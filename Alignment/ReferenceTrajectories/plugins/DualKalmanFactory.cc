/// \class DualKalmanFactory
///
/// A factory that produces reference trajectory instances of 
/// class DualKalmanTrajectory from a given TrajTrackPairCollection.
///
/// Currently two methods to set residual and error can be choosen via cfg:
///   1: the unbiased residal approach
///   2: the pull approach
///
///  \author    : Gero Flucke
///  date       : October 2008
///  $Revision: 1.3 $
///  $Date: 2010/09/10 13:25:45 $
///  (last update by $Author: mussgill $)

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"
// #include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "Alignment/ReferenceTrajectories/interface/DualKalmanTrajectory.h" 

#include <algorithm>


class DualKalmanFactory : public TrajectoryFactoryBase
{

public:

  DualKalmanFactory( const edm::ParameterSet & config );
  virtual ~DualKalmanFactory();

  /// Produce the reference trajectories.
  virtual const ReferenceTrajectoryCollection 
  trajectories(const edm::EventSetup &setup, const ConstTrajTrackPairCollection &tracks,
	       const reco::BeamSpot &beamSpot) const;

  virtual const ReferenceTrajectoryCollection 
  trajectories(const edm::EventSetup &setup, const ConstTrajTrackPairCollection &tracks,
	       const ExternalPredictionCollection &external, const reco::BeamSpot &beamSpot) const;

  virtual DualKalmanFactory* clone() const { return new DualKalmanFactory(*this); }

protected:

  struct DualKalmanInput 
  {
    TrajectoryStateOnSurface refTsos;
    Trajectory::DataContainer trajMeasurements;
    std::vector<unsigned int> fwdRecHitNums;
    std::vector<unsigned int> bwdRecHitNums;
  };

  const DualKalmanInput referenceStateAndRecHits(const ConstTrajTrackPair &track) const;

//   const TrajectoryStateOnSurface propagateExternal(const TrajectoryStateOnSurface &external,
// 						   const Surface &surface,
// 						   const MagneticField *magField) const;
  
  const double theMass;
  const int theResidMethod;
};


//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
DualKalmanFactory::DualKalmanFactory(const edm::ParameterSet &config) 
  : TrajectoryFactoryBase(config), theMass(config.getParameter<double>("ParticleMass")),
    theResidMethod(config.getParameter<int>("ResidualMethod"))
{
  // Since theResidMethod is passed to DualKalmanTrajectory, valid values are checked there.
  edm::LogInfo("Alignment") << "@SUB=DualKalmanFactory" << "Factory created.";
}

 
//-----------------------------------------------------------------------------------------------
DualKalmanFactory::~DualKalmanFactory() {}


//-----------------------------------------------------------------------------------------------
const DualKalmanFactory::ReferenceTrajectoryCollection
DualKalmanFactory::trajectories(const edm::EventSetup &setup,
				const ConstTrajTrackPairCollection &tracks,
				const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::ESHandle<MagneticField> magneticField;
  setup.get<IdealMagneticFieldRecord>().get(magneticField);

  ConstTrajTrackPairCollection::const_iterator itTracks = tracks.begin();

  while (itTracks != tracks.end()) { 
    const DualKalmanInput input = this->referenceStateAndRecHits(*itTracks);
    // Check input: If all hits were rejected, the TSOS is initialized as invalid.
    if (input.refTsos.isValid()) {
      ReferenceTrajectoryPtr ptr(new DualKalmanTrajectory(input.trajMeasurements,
							  input.refTsos,
							  input.fwdRecHitNums,
							  input.bwdRecHitNums,
							  magneticField.product(),
							  this->materialEffects(),
							  this->propagationDirection(),
							  theMass, theUseBeamSpot, beamSpot,
							  theResidMethod));
      trajectories.push_back(ptr);
    }
    ++itTracks;
  }

  return trajectories;
}

//-----------------------------------------------------------------------------------------------
const DualKalmanFactory::ReferenceTrajectoryCollection
DualKalmanFactory::trajectories(const edm::EventSetup &setup,
				const ConstTrajTrackPairCollection &tracks,
				const ExternalPredictionCollection &external,
				const reco::BeamSpot &beamSpot) const
{
  ReferenceTrajectoryCollection trajectories;

  edm::LogError("Alignment") << "@SUB=DualKalmanFactory::trajectories" 
			     << "Not implemented with ExternalPrediction.";
  return trajectories;
}


//-----------------------------------------------------------------------------------------------
const DualKalmanFactory::DualKalmanInput
DualKalmanFactory::referenceStateAndRecHits(const ConstTrajTrackPair &track) const
{
  // Same idea as in DualTrajectoryFactory::referenceStateAndRecHits(..):
  // Split trajectory in the middle, take middle as reference and provide first
  // and second half of hits, each starting from this middle hit.
  // In contrast to DualTrajectoryFactory we deal here with indices and not rechits directly
  // to be able to get measurements and uncertainties later from the Trajectory that is
  // provided by the Kalman track fit.

  DualKalmanInput input;
 
  // get the trajectory measurements in the correct order, i.e. reverse if needed
  input.trajMeasurements = this->orderedTrajectoryMeasurements(*track.first);

  // get indices of relevant trajectory measurements to find middle of them
  std::vector<unsigned int> usedTrajMeasNums;
  for (unsigned int iM = 0; iM < input.trajMeasurements.size(); ++iM) {
    if (this->useRecHit(input.trajMeasurements[iM].recHit())) usedTrajMeasNums.push_back(iM);
  }
  unsigned int nRefStateMeas = usedTrajMeasNums.size()/2;

  // get the valid RecHits numbers
  for (unsigned int iMeas = 0; iMeas < usedTrajMeasNums.size(); ++iMeas) {
    if (iMeas < nRefStateMeas) {
      input.bwdRecHitNums.push_back(usedTrajMeasNums[iMeas]);
    } else if (iMeas > nRefStateMeas) {
      input.fwdRecHitNums.push_back(usedTrajMeasNums[iMeas]);
    } else { // iMeas == nRefStateMeas
      if (input.trajMeasurements[usedTrajMeasNums[iMeas]].updatedState().isValid()) {
	input.refTsos = input.trajMeasurements[usedTrajMeasNums[iMeas]].updatedState();
	input.bwdRecHitNums.push_back(usedTrajMeasNums[iMeas]);
	input.fwdRecHitNums.push_back(usedTrajMeasNums[iMeas]);
      } else {
	// if the hit/tsos of the middle hit is not valid, try the next one...
	++nRefStateMeas; // but keep hit if only TSOS bad
	input.bwdRecHitNums.push_back(usedTrajMeasNums[iMeas]);
      }
    }
  }

  // bring input.fwdRecHits into correct order
  std::reverse(input.bwdRecHitNums.begin(), input.bwdRecHitNums.end());

  return input;
}

// //-----------------------------------------------------------------------------------------------
// const TrajectoryStateOnSurface
// DualKalmanFactory::propagateExternal(const TrajectoryStateOnSurface &external,
// 				     const Surface &surface,
// 				     const MagneticField *magField) const
// {
//   AnalyticalPropagator propagator(magField, anyDirection);
//   const std::pair<TrajectoryStateOnSurface, double> tsosWithPath =
//     propagator.propagateWithPath(external, surface);
//   return tsosWithPath.first;
//}


DEFINE_EDM_PLUGIN( TrajectoryFactoryPlugin, DualKalmanFactory, "DualKalmanFactory" );
