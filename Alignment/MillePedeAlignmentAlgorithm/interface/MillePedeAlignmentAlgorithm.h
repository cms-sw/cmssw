#ifndef Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h
#define Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h

/// \class MillePedeAlignmentAlgorithm
///
///  CMSSW interface to pede: produces pede's binary input and steering
///
///  \author    : Gero Flucke
///  date       : October 2006
///  $Revision: 1.1 $
///  $Date: 2006/10/20 13:44:13 $
///  (last update by $Author: flucke $)


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryBase.h"


#include <vector>

class MagneticField;
class MillePedeMonitor;
class PedeSteerer;
class Mille;

class MillePedeAlignmentAlgorithm : public AlignmentAlgorithmBase
{

 public:
  
  /// Constructor
  MillePedeAlignmentAlgorithm(const edm::ParameterSet &cfg);

  /// Destructor
  virtual ~MillePedeAlignmentAlgorithm();

  /// Call at beginning of job
  virtual void initialize(const edm::EventSetup &setup, AlignableTracker *tracker, 
		  AlignmentParameterStore *store);

  /// Call at end of job
  virtual void terminate();

  /// Run the algorithm on trajectories and tracks
  virtual void run(const edm::EventSetup &setup, const TrajTrackPairCollection &tracks);

 private:
  enum MeasurementDirection {kLocalX = 0, kLocalY};

  ReferenceTrajectoryBase::ReferenceTrajectoryPtr
    referenceTrajectory(const TrajectoryStateOnSurface &refTsos,
			const TransientTrackingRecHit::ConstRecHitContainer &hitVec,
			const MagneticField *magField) const;
  /// If hit is usable: callMille for x and (probably) y direction,
  /// return value as globalDerivatives, merged for x and y.
  int addGlobalDerivatives(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
			   unsigned int iHit);
  /// -1: problem, 0: no hit on considered alignable, 1: OK
  int globalDerivatives(const TransientTrackingRecHit::ConstRecHitPointer &recHit,
			const TrajectoryStateOnSurface &tsos, MeasurementDirection xOrY,
			std::vector<float> &globalDerivatives, 
			std::vector<int> &globalLabels) const;
  void callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, 
		 unsigned int iTrajHit, MeasurementDirection xOrY,
		 const std::vector<float> &globalDerivatives,
		 const std::vector<int> &globalLabels);

  bool is2D(const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;
  const MagneticField* getMagneticField(const edm::EventSetup& setup); //const;

  void recursiveFillLabelHist(Alignable *ali) const; // temporary?
  bool readFromPede(const std::string &pedeOutFile);

  edm::ParameterSet         theConfig;
  AlignmentParameterStore  *theAlignmentParameterStore;
  std::vector<Alignable*>   theAlignables;
  AlignableNavigator       *theAlignableNavigator;
  MillePedeMonitor         *theMonitor;
  Mille                    *theMille;
  PedeSteerer              *thePedeSteer;
  int                       theMinNumHits;

  std::vector<float>        theFloatBuffer;
  std::vector<int>          theIntBuffer;
};

#endif
