#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"
#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

#include "BzeroReferenceTrajectoryFactory.h"

/// A factory that produces instances of class ReferenceTrajectory from a given TrajTrackPairCollection.
/// If |B| = 0 T and configuration parameter UseBzeroIfFieldOff is True,
/// hand-over to the BzeroReferenceTrajectoryFactory.

class ReferenceTrajectoryFactory : public TrajectoryFactoryBase {
public:
  ReferenceTrajectoryFactory(const edm::ParameterSet &config, edm::ConsumesCollector &iC);
  ~ReferenceTrajectoryFactory() override;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_MagFieldToken;

  /// Produce the reference trajectories.
  const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
                                                   const ConstTrajTrackPairCollection &tracks,
                                                   const reco::BeamSpot &beamSpot) const override;

  const ReferenceTrajectoryCollection trajectories(const edm::EventSetup &setup,
                                                   const ConstTrajTrackPairCollection &tracks,
                                                   const ExternalPredictionCollection &external,
                                                   const reco::BeamSpot &beamSpot) const override;

  ReferenceTrajectoryFactory *clone() const override { return new ReferenceTrajectoryFactory(*this); }

protected:
  ReferenceTrajectoryFactory(const ReferenceTrajectoryFactory &other);
  const TrajectoryFactoryBase *bzeroFactory() const;
  const TrajectoryFactoryBase *bzeroFactory(edm::ConsumesCollector &iC) const;

  double theMass;
  bool theUseBzeroIfFieldOff;
  //edm::ParameterSet pset;
  mutable const TrajectoryFactoryBase *theBzeroFactory;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

ReferenceTrajectoryFactory::ReferenceTrajectoryFactory(const edm::ParameterSet &config, edm::ConsumesCollector &iC)
    : TrajectoryFactoryBase(config, iC),
      m_MagFieldToken(iC.esConsumes()),
      theMass(config.getParameter<double>("ParticleMass")),
      theUseBzeroIfFieldOff(config.getParameter<bool>("UseBzeroIfFieldOff")),
      theBzeroFactory(nullptr) {
  edm::LogInfo("Alignment") << "@SUB=ReferenceTrajectoryFactory"
                            << "mass: " << theMass
                            << "\nusing Bzero if |B| = 0: " << (theUseBzeroIfFieldOff ? "yes" : "no");
  // We take the config of this factory, copy it, replace its name and add
  // the momentum parameter as expected by BzeroReferenceTrajectoryFactory and create it:
  //
  edm::ParameterSet pset;
  pset.copyForModify(config);
  // next two lines not needed, but may help to better understand log file:
  pset.eraseSimpleParameter("TrajectoryFactoryName");
  pset.addParameter("TrajectoryFactoryName", std::string("BzeroReferenceTrajectoryFactory"));
  pset.addParameter("MomentumEstimate", config.getParameter<double>("MomentumEstimateFieldOff"));
  theBzeroFactory = new BzeroReferenceTrajectoryFactory(pset, iC);
}

ReferenceTrajectoryFactory::ReferenceTrajectoryFactory(const ReferenceTrajectoryFactory &other)
    : TrajectoryFactoryBase(other),
      theMass(other.theMass),
      theUseBzeroIfFieldOff(other.theUseBzeroIfFieldOff),
      theBzeroFactory(nullptr)  // copy data members, but no double pointing to same Bzero factory...
{}

ReferenceTrajectoryFactory::~ReferenceTrajectoryFactory(void) { delete theBzeroFactory; }

const ReferenceTrajectoryFactory::ReferenceTrajectoryCollection ReferenceTrajectoryFactory::trajectories(
    const edm::EventSetup &setup, const ConstTrajTrackPairCollection &tracks, const reco::BeamSpot &beamSpot) const {
  const MagneticField *magneticField = &setup.getData(m_MagFieldToken);

  if (theUseBzeroIfFieldOff && magneticField->inTesla(GlobalPoint(0., 0., 0.)).mag2() < 1.e-6) {
    return this->bzeroFactory()->trajectories(setup, tracks, beamSpot);
  }

  ReferenceTrajectoryCollection trajectories;

  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();

  while (itTracks != tracks.end()) {
    TrajectoryInput input = this->innermostStateAndRecHits(*itTracks);

    // Check input: If all hits were rejected, the TSOS is initialized as invalid.
    if (input.first.isValid()) {
      ReferenceTrajectoryBase::Config config(materialEffects(), propagationDirection(), theMass);
      config.useBeamSpot = useBeamSpot_;
      config.includeAPEs = includeAPEs_;
      config.allowZeroMaterial = allowZeroMaterial_;
      // set the flag for reversing the RecHits to false, since they are already in the correct order.
      config.hitsAreReverse = false;
      trajectories.push_back(
          ReferenceTrajectoryPtr(new ReferenceTrajectory(input.first, input.second, magneticField, beamSpot, config)));
    }

    ++itTracks;
  }

  return trajectories;
}

const ReferenceTrajectoryFactory::ReferenceTrajectoryCollection ReferenceTrajectoryFactory::trajectories(
    const edm::EventSetup &setup,
    const ConstTrajTrackPairCollection &tracks,
    const ExternalPredictionCollection &external,
    const reco::BeamSpot &beamSpot) const {
  ReferenceTrajectoryCollection trajectories;

  if (tracks.size() != external.size()) {
    edm::LogInfo("ReferenceTrajectories")
        << "@SUB=ReferenceTrajectoryFactory::trajectories"
        << "Inconsistent input:\n"
        << "\tnumber of tracks = " << tracks.size() << "\tnumber of external predictions = " << external.size();
    return trajectories;
  }
  const MagneticField *magneticField = &setup.getData(m_MagFieldToken);

  if (theUseBzeroIfFieldOff && magneticField->inTesla(GlobalPoint(0., 0., 0.)).mag2() < 1.e-6) {
    return this->bzeroFactory()->trajectories(setup, tracks, external, beamSpot);
  }

  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();
  ExternalPredictionCollection::const_iterator itExternal = external.begin();

  while (itTracks != tracks.end()) {
    TrajectoryInput input = innermostStateAndRecHits(*itTracks);
    // Check input: If all hits were rejected, the TSOS is initialized as invalid.
    if (input.first.isValid()) {
      if ((*itExternal).isValid() && sameSurface((*itExternal).surface(), input.first.surface())) {
        ReferenceTrajectoryBase::Config config(materialEffects(), propagationDirection(), theMass);
        config.useBeamSpot = useBeamSpot_;
        config.includeAPEs = includeAPEs_;
        config.allowZeroMaterial = allowZeroMaterial_;
        // set the flag for reversing the RecHits to false, since they are already in the correct order.
        config.hitsAreReverse = false;
        ReferenceTrajectoryPtr refTraj(
            new ReferenceTrajectory(*itExternal, input.second, magneticField, beamSpot, config));

        AlgebraicSymMatrix externalParamErrors(asHepMatrix<5>((*itExternal).localError().matrix()));
        refTraj->setParameterErrors(externalParamErrors);
        trajectories.push_back(refTraj);
      } else {
        ReferenceTrajectoryBase::Config config(materialEffects(), propagationDirection(), theMass);
        config.useBeamSpot = useBeamSpot_;
        config.includeAPEs = includeAPEs_;
        config.allowZeroMaterial = allowZeroMaterial_;
        config.hitsAreReverse = false;
        trajectories.push_back(ReferenceTrajectoryPtr(
            new ReferenceTrajectory(input.first, input.second, magneticField, beamSpot, config)));
      }
    }

    ++itTracks;
    ++itExternal;
  }

  return trajectories;
}

const TrajectoryFactoryBase *ReferenceTrajectoryFactory::bzeroFactory() const {
  if (!theBzeroFactory) {
    edm::LogInfo("Alignment") << "@SUB=ReferenceTrajectoryFactory::bzeroFactory"
                              << "Using BzeroReferenceTrajectoryFactory for some (all?) events.";
  }
  return theBzeroFactory;
}

DEFINE_EDM_PLUGIN(TrajectoryFactoryPlugin, ReferenceTrajectoryFactory, "ReferenceTrajectoryFactory");
