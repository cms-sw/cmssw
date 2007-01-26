#ifndef Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h
#define Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h

/// \class MillePedeAlignmentAlgorithm
///
///  CMSSW interface to pede: produces pede's binary input and steering
///
///  \author    : Gero Flucke
///  date       : October 2006
///  $Revision: 1.4 $
///  $Date: 2007/01/25 11:04:58 $
///  (last update by $Author: flucke $)


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectoryBase.h"


#include <vector>
#include <string>

class AlignableTracker;
class AlignableMuon;

class MagneticField;

class AlignmentParameters;

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
			  AlignableMuon *muon, AlignmentParameterStore *store);

  /// Call at end of job
  virtual void terminate();

  /// Run the algorithm on trajectories and tracks
  virtual void run(const edm::EventSetup &setup, const ConstTrajTrackPairCollection &tracks);

 private:
  enum MeasurementDirection {kLocalX = 0, kLocalY};

  ReferenceTrajectoryBase::ReferenceTrajectoryPtr
    referenceTrajectory(const TrajectoryStateOnSurface &refTsos,
			const TransientTrackingRecHit::ConstRecHitContainer &hitVec,
			const MagneticField *magField) const;
  /// If hit is usable: callMille for x and (probably) y direction.
  /// If globalDerivatives fine: returns 2 if 2D-hit and 1 if 1D-hit. Returns -1 if any problem.
  /// (for params cf. globalDerivatives)
  int addGlobalDerivatives(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
			   unsigned int iHit, const TrajectoryStateOnSurface &trackTsos,
			   AlignmentParameters *&params);
  /// -1: problem, 0: no hit on considered alignable, 1: OK
  /// if OK: params filled with pointer to parameters of the alignable which is hit
  int globalDerivatives(const TransientTrackingRecHit::ConstRecHitPointer &recHit,
			const TrajectoryStateOnSurface &tsos, MeasurementDirection xOrY,
			std::vector<float> &globalDerivatives, std::vector<int> &globalLabels,
			AlignmentParameters *&params) const;
  void callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, 
		 unsigned int iTrajHit, MeasurementDirection xOrY,
		 const std::vector<float> &globalDerivatives, const std::vector<int> &globalLabels);
  /// true if hit belongs to 2D detector (currently tracker specific)
  bool is2D(const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;
  const MagneticField* getMagneticField(const edm::EventSetup& setup) const;

  bool readFromPede(const std::string &pedeOutFile);
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
  void orderedTsos(const Trajectory *traj, 
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
  int                       theMinNumHits;
  bool                      theUseTrackTsos;

  std::vector<float>        theFloatBuffer;
  std::vector<int>          theIntBuffer;
};

#endif
