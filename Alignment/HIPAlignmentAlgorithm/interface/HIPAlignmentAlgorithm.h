#ifndef Alignment_HIPAlignmentAlgorithm_HIPAlignmentAlgorithm_h
#define Alignment_HIPAlignmentAlgorithm_HIPAlignmentAlgorithm_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Riostream.h"

#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h" 	 
#include "DataFormats/Alignment/interface/AliClusterValueMap.h" 	 
#include "Utilities/General/interface/ClassName.h" 	 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h" 	 
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h" 	 
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h" 	 

class AlignableNavigator;
class TFile;
class TTree;

class HIPAlignmentAlgorithm : public AlignmentAlgorithmBase
{

 public:
  
  /// Constructor
  HIPAlignmentAlgorithm(const edm::ParameterSet& cfg);

  /// Destructor
  ~HIPAlignmentAlgorithm() {};

  /// Call at beginning of job
  void initialize( const edm::EventSetup& setup, 
                   AlignableTracker* tracker, AlignableMuon* muon, AlignableExtras* extras, 
                   AlignmentParameterStore* store);

  /// Call at end of job
  void terminate(const edm::EventSetup& setup);

  /// Called at start of new loop
  void startNewLoop(void);

  /// Run the algorithm
  void run(const edm::EventSetup& setup, const EventInfo& eventInfo);

 private:

  // private member functions
  
  bool processHit1D(const AlignableDetOrUnitPtr& alidet,
		    const Alignable* ali,
		    const TrajectoryStateOnSurface & tsos,
		    const TrackingRecHit* hit,
                    double hitwt);

  bool processHit2D(const AlignableDetOrUnitPtr& alidet,
		    const Alignable* ali,
		    const TrajectoryStateOnSurface & tsos,
		    const TrackingRecHit* hit,
                    double hitwt);  

  int readIterationFile(std::string filename);
  void writeIterationFile(std::string filename, int iter);
  void setAlignmentPositionError(void);
  double calcAPE(double* par, int iter, double function);
  void bookRoot(void);
  void fillRoot(const edm::EventSetup& setup);
  bool calcParameters(Alignable* ali,int setDet, double start, double step);
  void collector(void);
  int  fillEventwiseTree(const char *filename, int iter, int ierr);
  // private data members

  std::unique_ptr<AlignableObjectId> alignableObjectId_;
  AlignmentParameterStore* theAlignmentParameterStore;
  std::vector<Alignable*> theAlignables;
  AlignableNavigator* theAlignableDetAccessor;

  AlignmentIORoot theIO;
  int ioerr;
  int theIteration;

  // steering parameters

  // verbosity flag
  bool verbose;
  // names of IO root files
  std::string outfile,outfile2,outpath,suvarfile,sparameterfile;
  std::string struefile,smisalignedfile,salignedfile,siterationfile,ssurveyfile;

  // alignment position error parameters
  bool theApplyAPE;
  bool themultiIOV;
  std::vector<edm::ParameterSet> theAPEParameterSet;
	std::vector<unsigned> theIOVrangeSet;
  std::vector<std::pair<std::vector<Alignable*>, std::vector<double> > > theAPEParameters;
  // max allowed pull (residual / uncertainty) on a hit used in alignment
  double theMaxAllowedHitPull;
  // min number of hits on alignable to calc parameters
  int theMinimumNumberOfHits;
  // max allowed rel error on parameter (else not used)
  double theMaxRelParameterError;
  // collector mode (parallel processing)
  bool isCollector;
  int theCollectorNJobs;
  std::string theCollectorPath;
  int theEventPrescale,theCurrentPrescale;
  bool trackPs,trackWt,IsCollision,uniEta;
  double Scale,cos_cut,col_cut;
  bool theFillTrackMonitoring;
  std::vector<double> SetScanDet;

  const std::vector<std::string> surveyResiduals_;
  std::vector<align::StructureType> theLevels; // for survey residuals

  // root tree variables
  TFile* theFile;
  TTree* theTree; // event-wise tree
  TTree* hitTree; // hit-wise tree
  TFile* theFile2;
  TTree* theTree2; // alignable-wise tree
  TFile* theFile3;
  TTree* theTree3; // survey tree

  // variables for event-wise tree
  static const int MAXREC = 99;
  //int m_Run,m_Event;
  int m_Ntracks,m_Nhits[MAXREC],m_nhPXB[MAXREC],m_nhPXF[MAXREC],m_nhTIB[MAXREC],m_nhTOB[MAXREC],m_nhTID[MAXREC],m_nhTEC[MAXREC];
  float m_Pt[MAXREC],m_Eta[MAXREC],m_Phi[MAXREC],m_Chi2n[MAXREC],m_P[MAXREC],m_d0[MAXREC],m_dz[MAXREC],m_wt[MAXREC];

  // variables for hit-wise tree
  float m_sinTheta,m_hitwt,m_angle;
  align::ID m_detId;

  // variables for alignable-wise tree
  int m2_Nhit,m2_Type,m2_Layer;
  float m2_Xpos, m2_Ypos, m2_Zpos, m2_Eta, m2_Phi; 
  align::ID m2_Id;
  align::StructureType m2_ObjId;

  // variables for survey tree 
  align::ID m3_Id;
  align::StructureType m3_ObjId;
  float m3_par[6];
};

#endif
