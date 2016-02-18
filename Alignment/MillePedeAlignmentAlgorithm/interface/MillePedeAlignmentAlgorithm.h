#ifndef Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h
#define Alignment_MillePedeAlignmentAlgorithm_MillePedeAlignmentAlgorithm_h

/// \class MillePedeAlignmentAlgorithm
///
///  CMSSW interface to pede: produces pede's binary input and steering file(s)
///
///  \author    : Gero Flucke
///  date       : October 2006
///  $Revision: 1.36 $
///  $Date: 2012/08/10 09:01:11 $
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
class AlignableExtras;

class AlignmentParameters;
class AlignableNavigator;
class AlignableDetOrUnitPtr;
class AlignmentUserVariables;

class AlignmentParameterStore;

class IntegratedCalibrationBase;

class MillePedeMonitor;
class PedeSteerer;
class PedeLabelerBase;
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

  /// Called at beginning of job
  virtual void initialize(const edm::EventSetup &setup,
			  AlignableTracker *tracker, AlignableMuon *muon, AlignableExtras *extras,
			  AlignmentParameterStore *store);

  /// Returns whether MP supports calibrations
  virtual bool supportsCalibrations() override;
  /// Pass integrated calibrations to Millepede (they are not owned by Millepede!)
  virtual bool addCalibrations(const std::vector<IntegratedCalibrationBase*> &iCals);

  /// Called at end of job
  virtual void terminate(const edm::EventSetup& iSetup);
  /// Called at end of job
  virtual void terminate();

  /// Returns whether MP should process events in the current configuration
  virtual bool processesEvents() override;
  /// Run the algorithm on trajectories and tracks
  virtual void run(const edm::EventSetup &setup, const EventInfo &eventInfo);

  // TODO: This method does NOT match endRun() in base class! Nobody is
  //       calling this?
  /// Run on run products, e.g. TkLAS
  virtual void endRun(const EventInfo&, const EndRunInfo&,
                      const edm::EventSetup&); //override;

  // This one will be called since it matches the interface of the base class
  virtual void endRun(const EndRunInfo &runInfo, const edm::EventSetup &setup);


/*   virtual void beginLuminosityBlock(const edm::EventSetup &setup) {} */
/*   virtual void endLuminosityBlock(const edm::EventSetup &setup) {} */

  /// Called in order to pass parameters to alignables for a specific run
  /// range in case the algorithm supports run range dependent alignment.
  virtual bool setParametersForRunRange(const RunRange &runrange);

 private:
  enum MeasurementDirection {kLocalX = 0, kLocalY};

  /// fill mille for a trajectory, returning number of x/y hits ([0,0] if 'bad' trajectory)
  std::pair<unsigned int, unsigned int>
    addReferenceTrajectory(const edm::EventSetup &setup, const EventInfo &eventInfo, 
			   const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr);

  /// If hit is usable: callMille for x and (probably) y direction.
  /// If globalDerivatives fine: returns 2 if 2D-hit, 1 if 1D-hit, 0 if no Alignable for hit.
  /// Returns -1 if any problem (for params cf. globalDerivativesHierarchy)
  int addMeasurementData(const edm::EventSetup &setup, const EventInfo &eventInfo, 
			 const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
			 unsigned int iHit, AlignmentParameters *&params);

  /// Add global data (labels, derivatives) to GBL trajectory
  /// Returns -1 if any problem (for params cf. globalDerivativesHierarchy)
  int addGlobalData(const edm::EventSetup &setup, const EventInfo &eventInfo,
                    const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
                    unsigned int iHit, gbl::GblPoint &gblPoint);

  /// Increase hit counting of MillePedeVariables behind each parVec[i]
  /// (and also for parameters higher in hierarchy),
  /// assuming 'parVec' and 'validHitVecY' to be parallel.
  /// Returns number of valid y-hits.
  unsigned int addHitCount(const std::vector<AlignmentParameters*> &parVec,
			   const std::vector<bool> &validHitVecY) const;

  /// adds data from reference trajectory from a specific Hit
  void addRefTrackData2D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
			 unsigned int iTrajHit, TMatrixDSym &aHitCovarianceM,
			 TMatrixF &aHitResidualsM, TMatrixF &aLocalDerivativesM);
  
  /// adds data for virtual measurements from reference trajectory
  void addVirtualMeas(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
		      unsigned int iVirtualMeas);
			   
 /// adds data for a specific virtual measurement from reference trajectory 
  void addRefTrackVirtualMeas1D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
				unsigned int iVirtualMeas,TMatrixDSym &aHitCovarianceM,
				TMatrixF &aHitResidualsM, TMatrixF &aLocalDerivativesM);
  
  /// recursively adding derivatives and labels, false if problems
  bool globalDerivativesHierarchy(const EventInfo &eventInfo, 
				  const TrajectoryStateOnSurface &tsos,
				  Alignable *ali, const AlignableDetOrUnitPtr &alidet,
				  std::vector<float> &globalDerivativesX,
				  std::vector<float> &globalDerivativesY,
				  std::vector<int> &globalLabels,
				  AlignmentParameters *&lowestParams) const;

  /// recursively adding derivatives (double) and labels, false if problems
  bool globalDerivativesHierarchy(const EventInfo &eventInfo,
                                  const TrajectoryStateOnSurface &tsos,
                                  Alignable *ali, const AlignableDetOrUnitPtr &alidet,
                                  std::vector<double> &globalDerivativesX,
                                  std::vector<double> &globalDerivativesY,
                                  std::vector<int> &globalLabels,
                                  AlignmentParameters *&lowestParams) const;

  /// adding derivatives from integrated calibrations
  void globalDerivativesCalibration(const TransientTrackingRecHit::ConstRecHitPointer &recHit,
                                    const TrajectoryStateOnSurface &tsos,
                                    const edm::EventSetup &setup, const EventInfo &eventInfo,
                                    std::vector<float> &globalDerivativesX,
                                    std::vector<float> &globalDerivativesY,
                                    std::vector<int> &globalLabels) const;

  /// calls callMille1D or callMille2D
  int callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
		unsigned int iTrajHit, const std::vector<int> &globalLabels,
		const std::vector<float> &globalDerivativesX,
		const std::vector<float> &globalDerivativesY);

  /// calls Mille for 1D hits
  int callMille1D(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
		  unsigned int iTrajHit, const std::vector<int> &globalLabels,
		  const std::vector<float> &globalDerivativesX);

  /// calls Mille for x and possibly y component of hit,
  /// y is skipped for non-real 2D (e.g. SiStripRecHit2D),
  /// for TID/TEC first diagonalises if correlation is larger than configurable
  int callMille2D ( const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr,
		    unsigned int iTrajHit, const std::vector<int> &globalLabels,
		    const std::vector<float> &globalDerivativesx,
		    const std::vector<float> &globalDerivativesy);
  void  diagonalize(TMatrixDSym &aHitCovarianceM, TMatrixF &aLocalDerivativesM,
		    TMatrixF &aHitResidualsM,TMatrixF &theGlobalDerivativesM) const;
  // deals with the non matrix format of theFloatBufferX ...
  void makeGlobDerivMatrix(const std::vector<float> &globalDerivativesx,
			   const std::vector<float> &globalDerivativesy,
			   TMatrixF &aGlobalDerivativesM);

//   void callMille(const ReferenceTrajectoryBase::ReferenceTrajectoryPtr &refTrajPtr, 
// 		 unsigned int iTrajHit, MeasurementDirection xOrY,
// 		 const std::vector<float> &globalDerivatives, const std::vector<int> &globalLabels);
  /// true if hit belongs to 2D detector (currently tracker specific)
  bool is2D(const TransientTrackingRecHit::ConstRecHitPointer &recHit) const;

  /// read pede input defined by 'psetName', flag to create/not create MillePedeVariables
  bool readFromPede(const edm::ParameterSet &mprespset, bool setUserVars,
		    const RunRange &runrange);
  bool areEmptyParams(const std::vector<Alignable*> &alignables) const;
  unsigned int doIO(int loop) const;
  /// add MillePedeVariables for each AlignmentParameters (exception if no parameters...)
  void buildUserVariables(const std::vector<Alignable*> &alignables) const;

  /// Generates list of files to read, given the list and dir from the configuration.
  /// This will automatically expand formatting directives, if they appear.
  std::vector<std::string> getExistingFormattedFiles(const std::vector<std::string>& plainFiles, const std::string& theDir);

  void addLaserData(const EventInfo &eventInfo, 
		    const TkFittedLasBeamCollection &tkLasBeams,
		    const TsosVectorCollection &tkLasBeamTsoses);
  void addLasBeam(const EventInfo &eventInfo, 
		  const TkFittedLasBeam &lasBeam,
		  const std::vector<TrajectoryStateOnSurface> &tsoses);

  /// add measurement data from PXB survey
  void addPxbSurvey(const edm::ParameterSet &pxbSurveyCfg);

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

  edm::ParameterSet         theConfig;
  unsigned int              theMode;
  std::string               theDir; /// directory for all kind of files
  AlignmentParameterStore  *theAlignmentParameterStore;
  std::vector<Alignable*>   theAlignables;
  AlignableNavigator       *theAlignableNavigator;
  MillePedeMonitor         *theMonitor;
  Mille                    *theMille;
  PedeLabelerBase          *thePedeLabels;
  PedeSteerer              *thePedeSteer;
  TrajectoryFactoryBase    *theTrajectoryFactory;
  std::vector<IntegratedCalibrationBase*> theCalibrations;
  unsigned int              theMinNumHits;
  double                    theMaximalCor2D; /// maximal correlation allowed for 2D hit in TID/TEC.
                                             /// If larger, the 2D measurement gets diagonalized!!!
  int                       theLastWrittenIov; // keeping track for output trees...
  std::vector<float>        theFloatBufferX;
  std::vector<float>        theFloatBufferY;
  std::vector<int>          theIntBuffer;
  bool                      theDoSurveyPixelBarrel;
  // CHK for GBL
  gbl::MilleBinary              *theBinary;
  bool                      theGblDoubleBinary;
};

#endif
