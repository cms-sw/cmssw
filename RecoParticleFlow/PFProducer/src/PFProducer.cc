#include "RecoParticleFlow/PFProducer/interface/PFProducer.h"

// #include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Surface/interface/BoundCylinder.h"
#include "Geometry/Surface/interface/TkRotation.h"
#include "Geometry/Surface/interface/SimpleCylinderBounds.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h" 
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"  
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"


using namespace std;
using namespace edm;

PFProducer::PFProducer(const edm::ParameterSet& iConfig) :
  conf_(iConfig), trackAlgo_(iConfig)
{
  edm::LogInfo("PFProducer") << "Constructor" << std::endl;

  // use configuration file to setup input/output collection names
  tcCollection_ = iConfig.getParameter<std::string>("TrackCandidateCollection");
  pfRecTrackCollection_ = iConfig.getParameter<std::string>("PFRecTrackCollection");

  // register your products
  produces<reco::PFRecTrackCollection>(pfRecTrackCollection_);

  // dummy... just to be able to run
  // produces<reco::PFRecHitCollection >();  
}


PFProducer::~PFProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.) 
}


void PFProducer::produce(edm::Event& iEvent, 
			 const edm::EventSetup& iSetup) 
{
  edm::LogInfo("PFProducer") << "Produce event: " << iEvent.id().event()
			     <<" in run " << iEvent.id().run() << std::endl;

  //
  // Create empty output collections
  //
  reco::PFRecTrackCollection outputPFRecTrackCollection;
  std::auto_ptr< reco::PFRecTrackCollection > pOutputPFRecTrackCollection(new reco::PFRecTrackCollection);
  
  //
  // Declare and get stuff to be retrieved from ES
  //
  LogDebug("PFProducer") << "get tracker geometry" << "\n";
  edm::ESHandle<TrackerGeometry> theG;
  iSetup.get<TrackerDigiGeometryRecord>().get(theG);
  LogDebug("PFProducer") << "get magnetic field" << "\n";
  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  LogDebug("PFProducer") << "get the trajectory fitter from the ES" << "\n";
  edm::ESHandle<TrajectoryFitter> theFitter;
  std::string fitterName = conf_.getParameter<std::string>("Fitter");   
  iSetup.get<TrackingComponentsRecord>().get(fitterName, theFitter);
  LogDebug("PFProducer") << "get the trajectory propagator from the ES" 
			 << "\n";
  edm::ESHandle<Propagator> thePropagator;
  std::string propagatorName = conf_.getParameter<std::string>("Propagator");
  iSetup.get<TrackingComponentsRecord>().get(propagatorName, thePropagator);
  LogDebug("PFProducer") << "get the TransientTrackingRecHitBuilder" << "\n";
  edm::ESHandle<TransientTrackingRecHitBuilder> theBuilder;
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  iSetup.get<TransientRecHitRecord>().get(builderName, theBuilder);

  //
  // Prepare propagation tools
  //
  const MagneticField * magField = theMF.product();
  AnalyticalPropagator propagator(magField, alongMomentum);
  ReferenceCountingPointer<Surface> ecalWall(new BoundCylinder(GlobalPoint(0.,0.,0.), TkRotation<float>(), SimpleCylinderBounds(130., 130., -500., 500.)));

  //
  // Get track candidates and create smoothed tracks
  // Temporary solution. Let's hope that in the future, Trajectory objects 
  // will become persistent.
  //
  AlgoProductCollection algoResults;
  try{
    LogDebug("PFProducer") << "get the TrackCandidateCollection"
			   << " from the event, source is " 
			   << tcCollection_ <<"\n";
    edm::Handle<TrackCandidateCollection> theTCCollection;
    iEvent.getByLabel(tcCollection_, theTCCollection);

    //
    //run the algorithm  
    //
    LogDebug("PFProducer") << "run the tracking algorithm" << "\n";
    trackAlgo_.runWithCandidate(theG.product(), theMF.product(), 
				*theTCCollection,
				theFitter.product(), thePropagator.product(), 
				theBuilder.product(), algoResults);
  } catch (cms::Exception& e) { 
    edm::LogError("PFProducer") << "cms::Exception caught : " 
				<< "cannot get collection " 
				<< tcCollection_ << "\n" << e << "\n";
  }

  //
  // Loop over smoothed tracks and fill PFRecTrack collection
  //
  for(AlgoProductCollection::iterator itTrack = algoResults.begin();
      itTrack != algoResults.end(); itTrack++) {
    Trajectory*  theTraj  = (*itTrack).first;
    reco::Track* theTrack = (*itTrack).second;
    reco::PFRecTrack track(theTrack->parameters().charge(), 
			   reco::PFRecTrack::KF);

    //
    // Loop over trajectory measurements
    //
    std::vector<TrajectoryMeasurement> measurements = theTraj->measurements();
    // Order measurements along momentum
    int iTrajFirst = 0;
    int iTrajLast  = measurements.size();
    int increment = +1;
    if (theTraj->direction() == oppositeToMomentum) {
      iTrajFirst = measurements.size() - 1;
      iTrajLast = -1;
      increment = -1;
    }
    for (int iTraj = iTrajFirst; iTraj != iTrajLast; iTraj += increment) {
      TrajectoryStateOnSurface tsos = measurements[iTraj].updatedState();
      GlobalPoint v  = tsos.globalParameters().position();
      GlobalVector p = tsos.globalParameters().momentum();
      math::XYZPoint  pos(v.x(), v.y(), v.z());       
      math::XYZTLorentzVector mom(p.x(), p.y(), p.z(), p.mag());
      unsigned int detId = 
	measurements[iTraj].recHit()->det()->geographicalId().rawId();
      reco::PFTrajectoryPoint trajPt(detId, reco::PFTrajectoryPoint::NLayers, 
				     pos, mom);
      track.addMeasurement(trajPt);
      LogDebug("PFProducer") << trajPt << "\n";
    }

    // Closest approach of the beamline
    math::XYZPoint posClosest(theTrack->x(), theTrack->y(), theTrack->z());
    math::XYZTLorentzVector momClosest(theTrack->px(), theTrack->py(), 
				       theTrack->pz(), theTrack->p());
    reco::PFTrajectoryPoint closestPt(0, 
				      reco::PFTrajectoryPoint::ClosestApproach,
				      posClosest, momClosest);
    track.setPoint(reco::PFTrajectoryPoint::ClosestApproach, closestPt);

    // Propagate track to ECAL
    TrajectoryStateOnSurface outerTSOS;
    if (theTraj->direction() == alongMomentum)
      outerTSOS = measurements[measurements.size() - 1].updatedState();
    else
      outerTSOS = measurements[0].updatedState();
    TrajectoryStateOnSurface ecalTSOS = 
      propagator.propagate(outerTSOS, *ecalWall);
    GlobalPoint vECAL  = ecalTSOS.globalParameters().position();
    GlobalVector pECAL = ecalTSOS.globalParameters().momentum();
    math::XYZPoint posECAL(vECAL.x(), vECAL.y(), vECAL.z());       
    math::XYZTLorentzVector momECAL(pECAL.x(), pECAL.y(), pECAL.z(), 
				    pECAL.mag());
    reco::PFTrajectoryPoint ecalPt(0, reco::PFTrajectoryPoint::ECALEntrance, 
				   posECAL, momECAL);
    track.setPoint(track.nTrajectoryMeasurements() +
		   reco::PFTrajectoryPoint::ECALEntrance, ecalPt);

    outputPFRecTrackCollection.push_back(track);
    LogDebug("PFProducer") << "Add a new PFRecTrack " << track << "\n";
  }

  //
  // Put the products in the event
  //
  edm::LogInfo("PFProducer") << " Put the PFRecTrackCollection of " 
			     << outputPFRecTrackCollection.size() 
			     << " candidates in the Event" << std::endl;
  pOutputPFRecTrackCollection->assign(outputPFRecTrackCollection.begin(),
				      outputPFRecTrackCollection.end());
  iEvent.put(pOutputPFRecTrackCollection, pfRecTrackCollection_);
}


//define this as a plug-in
DEFINE_FWK_MODULE(PFProducer)
