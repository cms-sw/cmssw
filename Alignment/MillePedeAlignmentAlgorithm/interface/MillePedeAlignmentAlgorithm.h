#ifndef Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h
#define Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h

/// \class MillePedeAlignmentAlgorithm
///
///  CMSSW interface to pede: produces pede's binary input and steering file(s)
///
///  \author    : Gero Flucke
///  date       : October 2006
///  $Revision: 1.13 $
///  $Date: 2007/07/13 16:27:06 $
///  (last update by $Author: flucke $)


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"


#include <vector>
#include <string>

class Alignable;
class AlignableTracker;
class AlignableMuon;

class AlignmentParameters;
class AlignableNavigator;
class AlignableDetOrUnitPtr;
class AlignmentUserVariables;

class AlignmentParameterStore;

class MillePedeMonitor;
class PedeSteerer;
class Mille;
class TrajectoryFactoryBase;

class MillePedeAlignmentAlgorithm : public AlignmentAlgorithmBase
{
 public:
  /// Constructor
  MillePedeAlignmentAlgorithm(const edm::ParameterSet &cfg);

  /// Destructor
  virtual ~MillePedeAlignmentAlgorithm();

  /// Call at beginning of job
  virtual void initialize(const edm::EventSetup &setup, AlignableTracker *tracker,
			  AlignableMuon *muon, AlignmentParameterStore *store);

  /// Call at end of job
  virtual void terminate();

  /// Run the algorithm on trajectories and tracks
  virtual void run(const edm::EventSetup &setup, const ConstTrajTrackPairCollection &tracks);

 private:
  enum MeasurementDirection {kLocalX = 0, kLocalY};

  /// If hit is usable: callMille for x and (probably) y direction.
  /// If globalDerivatives fine: returns 2 if 2D-hit, 1 if 1D-hit, 0 if no Alignable for hit.
  /// Returns -1 if any problem (for params cf. globalDerivativesHierarchy)
  int addGlobalDerivatives(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
			   unsigned int iHit, const TrajectoryStateOnSurface &trackTsos,
			   AlignmentParameters *&params);
  /// recursively adding derivatives and labels, false if problems
  bool globalDerivativesHierarchy(const TrajectoryStateOnSurface &tsos,
				  Alignable *ali, const AlignableDetOrUnitPtr &alidet, bool hit2D,
				  std::vector<float> &globalDerivativesX,
				  std::vector<float> &globalDerivativesY,
				  std::vector<int> &globalLabels,
				  AlignmentParameters *&lowestParams) const;
  void callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, 
		 unsigned int iTrajHit, MeasurementDirection xOrY,
		 const std::vector<float> &globalDerivatives, const std::vector<int> &globalLabels);
  /// true if hit belongs to 2D detector (currently tracker specific)
  bool is2D(const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;

  bool readFromPede();
  bool areEmptyParams(const std::vector<Alignable*> &alignables) const;
  unsigned int doIO(int loop) const;
  /// add MillePedeVariables for each AlignmentParameters (exception if no parameters...)
  void buildUserVariables(const std::vector<Alignable*> &alignables) const;

  enum EModeBit {myMilleBit = 1 << 0, myPedeRunBit = 1 << 1, myPedeSteerBit = 1 << 2,
		 myPedeReadBit = 1 << 3};
  unsigned int decodeMode(const std::string &mode) const;
  bool isMode(unsigned int testMode) const {return (theMode & testMode);}
  bool addHitStatistics(int fromLoop, const std::string &outFile,
			const std::vector<std::string> &inFiles) const;
  bool addHits(const std::vector<Alignable*> &alis,
	       const std::vector<AlignmentUserVariables*> &mpVars) const;
  bool orderedTsos(const Trajectory *traj, 
		   std::vector<TrajectoryStateOnSurface> &trackTsos) const;
  edm::ParameterSet         theConfig;
  unsigned int              theMode;
  std::string               theDir; /// directory for all kind of files
  AlignmentParameterStore  *theAlignmentParameterStore;
  std::vector<Alignable*>   theAlignables;
  AlignableNavigator       *theAlignableNavigator;
  MillePedeMonitor         *theMonitor;
  Mille                    *theMille;
  PedeSteerer              *thePedeSteer;
  TrajectoryFactoryBase    *theTrajectoryFactory;
  int                       theMinNumHits;
  bool                      theUseTrackTsos;

  std::vector<float>        theFloatBufferX;
  std::vector<float>        theFloatBufferY;
  std::vector<int>          theIntBuffer;
};

#endif
