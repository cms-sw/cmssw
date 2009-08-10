#ifndef Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h
#define Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h

/// \class MillePedeAlignmentAlgorithm
///
///  CMSSW interface to pede: produces pede's binary input and steering file(s)
///
///  \author    : Gero Flucke
///  date       : October 2006
///  $Revision: 1.22 $
///  $Date: 2009/06/23 10:25:55 $
///  (last update by $Author: flucke $)

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectoryBase.h"


#include <vector>
#include <string>

#include <TMatrixDSym.h>
#include <TMatrixD.h>
#include <TMatrixF.h>

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
class PedeLabeler;
class Mille;
class TrajectoryFactoryBase;

// already from base class - and forward declaration does not work since typedef!
/* class TkFittedLasBeamCollection; */
/* class TsosVectorCollection; */

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
  virtual void run(const edm::EventSetup &setup, const EventInfo &eventInfo);

/*   virtual void beginRun(const edm::EventSetup &setup) {} */
  /// Run on run products, e.g. TkLAS
  virtual void endRun(const EndRunInfo &runInfo, const edm::EventSetup &setup);

/*   virtual void beginLuminosityBlock(const edm::EventSetup &setup) {} */
/*   virtual void endLuminosityBlock(const edm::EventSetup &setup) {} */

 private:
  enum MeasurementDirection {kLocalX = 0, kLocalY};

  /// If hit is usable: callMille for x and (probably) y direction.
  /// If globalDerivatives fine: returns 2 if 2D-hit, 1 if 1D-hit, 0 if no Alignable for hit.
  /// Returns -1 if any problem (for params cf. globalDerivativesHierarchy)
  int addMeasurementData(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
			   unsigned int iHit, const TrajectoryStateOnSurface &trackTsos,
			   AlignmentParameters *&params);

 // adds data from reference trajectory from a specific Hit
  void addRefTrackData2D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, unsigned int iTrajHit,TMatrixDSym &aHitCovarianceM, TMatrixF &aHitResidualsM, TMatrixF &aLocalDerivativesM);


  /// recursively adding derivatives and labels, false if problems
  bool globalDerivativesHierarchy(const TrajectoryStateOnSurface &tsos,
				  Alignable *ali, const AlignableDetOrUnitPtr &alidet,
				  std::vector<float> &globalDerivativesX,
				  std::vector<float> &globalDerivativesY,
				  std::vector<int> &globalLabels,
				  AlignmentParameters *&lowestParams) const;

 // calls Mille and diagonalises the covariance matrx of a Hit if neccesary
  int callMille2D ( const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
		    unsigned int iTrajHit, const std::vector<int> &globalLabels,
		    const std::vector<float> &globalDerivativesx,
		    const std::vector<float> &globalDerivativesy);
  void  diagonalize(TMatrixDSym &aHitCovarianceM, TMatrixF &aLocalDerivativesM,
		    TMatrixF &aHitResidualsM,TMatrixF &theGlobalDerivativesM) const;
  // deals with the non matrix format of theFloatBufferX ...
  void makeGlobDerivMatrix(const std::vector<float> &globalDerivativesx, const std::vector<float> &globalDerivativesy,TMatrixF &aGlobalDerivativesM);

//   void callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, 
// 		 unsigned int iTrajHit, MeasurementDirection xOrY,
// 		 const std::vector<float> &globalDerivatives, const std::vector<int> &globalLabels);
  /// true if hit belongs to 2D detector (currently tracker specific)
  bool is2D(const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;

  /// read pede input defined by 'psetName', flag to create/not create MillePedeVariables
  bool readFromPede(const edm::ParameterSet &mprespset, bool setUserVars);
  bool areEmptyParams(const std::vector<Alignable*> &alignables) const;
  unsigned int doIO(int loop) const;
  /// add MillePedeVariables for each AlignmentParameters (exception if no parameters...)
  void buildUserVariables(const std::vector<Alignable*> &alignables) const;

  void addLaserData(const TkFittedLasBeamCollection &tkLasBeams,
		    const TsosVectorCollection &tkLasBeamTsoses);
  void addLasBeam(const TkFittedLasBeam &lasBeam,
		  const std::vector<TrajectoryStateOnSurface> &tsoses);

  //--------------------------------------------------------
  // Data members
  //--------------------------------------------------------
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
  const PedeLabeler        *thePedeLabels;
  PedeSteerer              *thePedeSteer;
  TrajectoryFactoryBase    *theTrajectoryFactory;
  int                       theMinNumHits;
  double                    theMaximalCor2D; /// maximal correlation allowed for 2D hits. If larger
                                              /// the 2D measurement gets diagonalized!!!
  bool                      theUseTrackTsos;

  std::vector<float>        theFloatBufferX;
  std::vector<float>        theFloatBufferY;
  std::vector<int>          theIntBuffer;
};

#endif
