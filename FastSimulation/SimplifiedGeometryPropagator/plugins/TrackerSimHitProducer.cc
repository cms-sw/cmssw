#include <vector>
#include <memory>

// framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/UniformEngine/interface/UniformMagneticField.h"

// tracking
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

// fastsim
#include "FastSimulation/TrajectoryManager/interface/InsideBoundsMeasurementEstimator.h" //TODO move this
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"

// data formats
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// other
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "CondFormats/External/interface/DetID.h"
#include "FWCore/Framework/interface/ProducerBase.h"


///////////////////////////////////////////////
// Author: L. Vanelderen, S. Kurz
// Date: 29 May 2017
//////////////////////////////////////////////////////////


typedef std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;

namespace fastsim
{
    //! Produces SimHits in the tracker layers.
    /*!
        Also assigns the energy loss of the particle (ionization) with the SimHit.
        Furthermore, SimHits from different SubModules have to be sorted by their occurance!
        \sa EnergyLoss
    */
    class TrackerSimHitProducer : public InteractionModel
    {
        public:
        //! Constructor.
        TrackerSimHitProducer(const std::string & name,const edm::ParameterSet & cfg);

        //! Default destructor.
        ~TrackerSimHitProducer() override{;}

        //! Perform the interaction.
        /*!
            \param particle The particle that interacts with the matter.
            \param layer The detector layer that interacts with the particle.
            \param secondaries Particles that are produced in the interaction (if any).
            \param random The Random Engine.
        */
        void interact(Particle & particle,const SimplifiedGeometry & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random) override;
        
        //! Register the SimHit collection.
        void registerProducts(edm::ProducerBase & producer) const override;

        //! Store the SimHit collection.
        void storeProducts(edm::Event & iEvent) override;

        //! Helper funtion to create the actual SimHit on a detector (sub-) module.
        /*!
            \param particle Representation of the particle's trajectory
            \param pdgId The pdgId of the particle
            \param layerThickness The thickness of the layer (in radLengths)
            \param eLoss The energy that particle deposited in the detector (lost via ionisation)
            \param simTrackId The SimTrack this hit should be assigned to
            \param detector The detector element that is hit
            \param refPos Reference position that is used to sort the hits if layer consists of sub-detectors
            \ return Returns the SimHit and the distance relative to refPos since hits have to be ordered (in time) afterwards.
        */
        std::pair<double, std::unique_ptr<PSimHit>> createHitOnDetector(const TrajectoryStateOnSurface & particle, int pdgId, double layerThickness, double eLoss, int simTrackId, const GeomDet & detector, GlobalPoint & refPos);
        
        private:
        const double onSurfaceTolerance_;  //!< Max distance between particle and active (sub-) module. Otherwise particle has to be propagated.
        std::unique_ptr<edm::PSimHitContainer> simHitContainer_;  //!< The SimHit.
        double minMomentum_;  //!< Set the minimal momentum of incoming particle
        bool doHitsFromInboundParticles_;  //!< If not set, incoming particles (negative speed relative to center of detector) don't create a SimHits since reconstruction anyways not possible
    };
}



fastsim::TrackerSimHitProducer::TrackerSimHitProducer(const std::string & name,const edm::ParameterSet & cfg)
    : fastsim::InteractionModel(name)
    , onSurfaceTolerance_(0.01)
    , simHitContainer_(new edm::PSimHitContainer)
{
    // Set the minimal momentum
    minMomentum_ = cfg.getParameter<double>("minMomentumCut");
    // - if not set, particles from outside the beampipe with a negative speed in R direction are propagated but no SimHits
    // - particle with positive (negative) z and negative (positive) speed in z direction: no SimHits
    // -> this is not neccesary since a track reconstruction is not possible in this case anyways
    doHitsFromInboundParticles_ = cfg.getParameter<bool>("doHitsFromInboundParticles");
}

void fastsim::TrackerSimHitProducer::registerProducts(edm::ProducerBase & producer) const
{
    producer.produces<edm::PSimHitContainer>("TrackerHits");
}

void fastsim::TrackerSimHitProducer::storeProducts(edm::Event & iEvent)
{
    iEvent.put(std::move(simHitContainer_), "TrackerHits");
    simHitContainer_.reset(new edm::PSimHitContainer);
}

void fastsim::TrackerSimHitProducer::interact(Particle & particle,const SimplifiedGeometry & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random)
{
    // the energy deposit in the layer
    double energyDeposit = particle.getEnergyDeposit();
    particle.setEnergyDeposit(0);

    //
    // don't save hits from particle that did a loop or are inbound (coming from the outer part of the tracker, going towards the center)
    //
    if(!doHitsFromInboundParticles_){
        if(particle.isLooper()){
            return;
        }    
        if(particle.momentum().X()*particle.position().X() + particle.momentum().Y()*particle.position().Y() < 0){
            particle.setLooper();
            return;
        }
        if(layer.isForward() && ((particle.position().Z() > 0 && particle.momentum().Z() < 0) || (particle.position().Z() < 0 && particle.momentum().Z() > 0))){
            particle.setLooper();
            return;
        }
    }

    //
    // check that layer has tracker modules
    //
    if(!layer.getDetLayer())
    {
    return;
    }

    //
    // no material
    //
    if(layer.getThickness(particle.position(), particle.momentum()) < 1E-10)
    {
    return;
    }

    //
    // only charged particles
    //
    if(particle.charge()==0)
    {
    return;
    }

    //
    // save hit only if momentum higher than threshold
    //
    if(particle.momentum().Perp2() < minMomentum_*minMomentum_)
    {
    return;
    }

    // create the trajectory of the particle
    UniformMagneticField magneticField(layer.getMagneticFieldZ(particle.position())); 
    GlobalPoint  position( particle.position().X(), particle.position().Y(), particle.position().Z());
    GlobalVector momentum( particle.momentum().Px(), particle.momentum().Py(), particle.momentum().Pz());
    auto plane = layer.getDetLayer()->surface().tangentPlane(position);
    TrajectoryStateOnSurface trajectory(GlobalTrajectoryParameters( position, momentum, TrackCharge( particle.charge()), &magneticField), *plane);
    
    // find detectors compatible with the particle's trajectory
    AnalyticalPropagator propagator(&magneticField, anyDirection);
    InsideBoundsMeasurementEstimator est;
    std::vector<DetWithState> compatibleDetectors = layer.getDetLayer()->compatibleDets(trajectory, propagator, est);

    ////////
    // You have to sort the simHits in the order they occur!
    ////////

    // The old algorithm (sorting by distance to IP) doesn't seem to make sense to me (what if particle moves inwards?)

    // Detector layers have to be sorted by proximity to particle.position
    // Propagate particle backwards a bit to make sure it's outside any components
    std::map<double, std::unique_ptr<PSimHit>> distAndHits;
    // Position relative to which the hits should be sorted
    GlobalPoint positionOutside(particle.position().x()-particle.momentum().x()/particle.momentum().P()*10.,
                                particle.position().y()-particle.momentum().y()/particle.momentum().P()*10.,
                                particle.position().z()-particle.momentum().z()/particle.momentum().P()*10.);

    // FastSim: cheat tracking -> assign hits to closest charged daughter if particle decays
    int pdgId = particle.pdgId();
    int simTrackId = particle.getMotherSimTrackIndex() >=0 ? particle.getMotherSimTrackIndex() : particle.simTrackIndex();

    // loop over the compatible detectors
    for (const auto & detectorWithState : compatibleDetectors)
    {
        const GeomDet & detector = *detectorWithState.first;
        const TrajectoryStateOnSurface & particleState = detectorWithState.second;

        // if the detector has no components
        if(detector.isLeaf())
        {
            std::pair<double, std::unique_ptr<PSimHit>> hitPair = createHitOnDetector(particleState, pdgId, layer.getThickness(particle.position()), energyDeposit, simTrackId, detector, positionOutside);
            if(hitPair.second){
                distAndHits.insert(distAndHits.end(), std::move(hitPair));
            }
        }
        else
        {
            // if the detector has components
            for(const auto component : detector.components())
            {
                std::pair<double, std::unique_ptr<PSimHit>> hitPair = createHitOnDetector(particleState, pdgId,layer.getThickness(particle.position()), energyDeposit, simTrackId, *component, positionOutside);            
                if(hitPair.second){
                    distAndHits.insert(distAndHits.end(), std::move(hitPair));
                }
            }
        }
    }

    // Fill simHitContainer
    for(std::map<double, std::unique_ptr<PSimHit>>::const_iterator it = distAndHits.begin(); it != distAndHits.end(); it++){
        simHitContainer_->push_back(*(it->second));
    }
    
}

// Also returns distance to simHit since hits have to be ordered (in time) afterwards. Necessary to do explicit copy of TrajectoryStateOnSurface particle (not call by reference)
std::pair<double, std::unique_ptr<PSimHit>> fastsim::TrackerSimHitProducer::createHitOnDetector(const TrajectoryStateOnSurface & particle, int pdgId, double layerThickness, double eLoss, int simTrackId, const GeomDet & detector, GlobalPoint & refPos)
{
    // determine position and momentum of particle in the coordinate system of the detector
    LocalPoint localPosition;
    LocalVector localMomentum;

    // if the particle is close enough, no further propagation is needed
    if(fabs(detector.toLocal(particle.globalPosition()).z()) < onSurfaceTolerance_) 
    {
       localPosition = particle.localPosition();
       localMomentum = particle.localMomentum();
    }
    // else, propagate 
    else 
    {
        // find crossing of particle with detector layer
        HelixArbitraryPlaneCrossing crossing(particle.globalPosition().basicVector(),
                              particle.globalMomentum().basicVector(),
                              particle.transverseCurvature(),
                              anyDirection);
        std::pair<bool,double> path = crossing.pathLength(detector.surface());
        // case propagation succeeds
        if (path.first)     
        {
            localPosition = detector.toLocal( GlobalPoint( crossing.position(path.second)));
            localMomentum = detector.toLocal( GlobalVector( crossing.direction(path.second)));
            localMomentum = localMomentum.unit() * particle.localMomentum().mag();
        }
        // case propagation fails
        else
        {
            return std::pair<double, std::unique_ptr<PSimHit>>(0, std::unique_ptr<PSimHit>(nullptr));
        }
    }

    // find entry and exit point of particle in detector
    const Plane& detectorPlane = detector.surface();
    double halfThick = 0.5 * detectorPlane.bounds().thickness();
    double pZ = localMomentum.z();
    LocalPoint entry = localPosition + (-halfThick / pZ) * localMomentum;
    LocalPoint exit = localPosition + (halfThick / pZ) * localMomentum;
    double tof = particle.globalPosition().mag() / fastsim::Constants::speedOfLight ; // in nanoseconds
    
    // make sure the simhit is physically on the module
    double boundX = detectorPlane.bounds().width() / 2.;
    double boundY = detectorPlane.bounds().length() / 2.;
    // Special treatment for TID and TEC trapeziodal modules
    unsigned subdet = DetId(detector.geographicalId()).subdetId(); 
    if (subdet == 4 || subdet == 6) 
    boundX *=  1. - localPosition.y() / detectorPlane.position().perp();
    if(fabs(localPosition.x()) > boundX  || fabs(localPosition.y()) > boundY )
    {
       return std::pair<double, std::unique_ptr<PSimHit>>(0, std::unique_ptr<PSimHit>(nullptr));
    }

    // Create the hit: the energy loss rescaled to the module thickness
    // Total thickness is in radiation lengths, 1 radlen = 9.36 cm
    // Sensitive module thickness is about 30 microns larger than 
    // the module thickness itself
    eLoss *= (2. * halfThick - 0.003) / (9.36 * layerThickness);

    // Position of the hit in global coordinates
    GlobalPoint hitPos(detector.surface().toGlobal(localPosition));

    return std::pair<double, std::unique_ptr<PSimHit>>((hitPos-refPos).mag(),
                                        std::unique_ptr<PSimHit>(new PSimHit(entry, exit, localMomentum.mag(), tof, eLoss, pdgId,
                                                   detector.geographicalId().rawId(), simTrackId,
                                                   localMomentum.theta(),
                                                   localMomentum.phi())));
}

DEFINE_EDM_PLUGIN(
    fastsim::InteractionModelFactory,
    fastsim::TrackerSimHitProducer,
    "fastsim::TrackerSimHitProducer"
    );