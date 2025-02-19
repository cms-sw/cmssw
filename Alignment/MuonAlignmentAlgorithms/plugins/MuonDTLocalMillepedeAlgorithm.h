#ifndef Alignment_MuonAlignmentAlgorithms_MuonDTLocalMillepedeAlgorithm_h
#define Alignment_MuonAlignmentAlgorithms_MuonDTLocalMillepedeAlgorithm_h


#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "TH1D.h"
#include "TProfile.h"
#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
#include "TMatrixD.h"


class AlignableNavigator;
class TFile;
class TTree;
class AlignableDet;

#define MAX_HIT 60
#define MAX_HIT_CHAM 14
#define MAX_SEGMENT 5


class MuonDTLocalMillepedeAlgorithm : public AlignmentAlgorithmBase
{

 public:

  /// Constructor
  MuonDTLocalMillepedeAlgorithm(const edm::ParameterSet& cfg);

  /// Destructor
  ~MuonDTLocalMillepedeAlgorithm() {};

  /// Call at beginning of job
  void initialize( const edm::EventSetup& setup, 
                   AlignableTracker* tracker, AlignableMuon* muon, 
                   AlignmentParameterStore* store);

  /// Call at end of job
  void terminate(void);


  /// Run the algorithm on trajectories and tracks
  void run(const edm::EventSetup& setup, const EventInfo &eventInfo);
  //void run( const edm::EventSetup& , const ConstTrajTrackPairCollection& );



 private:

  // Builds the 4D segments  
  bool build4DSegments();

  // Declares the tree structure and associated the variables
  void setBranchTrees();
  
  //Auxiliar structure for 4D segment construction
  typedef struct {
    int nhits;
    float xc[MAX_HIT]; float yc[MAX_HIT]; float zc[MAX_HIT]; 
    float erx[MAX_HIT];
    int wh[MAX_HIT]; int st[MAX_HIT]; int sr[MAX_HIT];
    int sl[MAX_HIT]; int la[MAX_HIT];
  } Info1D;
  
  Info1D myTrack1D;
 
  
  //Block of variables for the tree 
  //---------------------------------------------------------
  float p, pt, eta, phi, charge;
  int nseg;
  int nphihits[MAX_SEGMENT];
  int nthetahits[MAX_SEGMENT];
  int nhits[MAX_SEGMENT];
  float xSl[MAX_SEGMENT]; 
  float dxdzSl[MAX_SEGMENT]; 
  float exSl[MAX_SEGMENT]; 
  float edxdzSl[MAX_SEGMENT]; 
  float exdxdzSl[MAX_SEGMENT]; 
  float ySl[MAX_SEGMENT]; 
  float dydzSl[MAX_SEGMENT]; 
  float eySl[MAX_SEGMENT]; 
  float edydzSl[MAX_SEGMENT]; 
  float eydydzSl[MAX_SEGMENT]; 
  float xSlSL1[MAX_SEGMENT]; 
  float dxdzSlSL1[MAX_SEGMENT]; 
  float exSlSL1[MAX_SEGMENT]; 
  float edxdzSlSL1[MAX_SEGMENT]; 
  float exdxdzSlSL1[MAX_SEGMENT]; 
  float xSL1SL3[MAX_SEGMENT]; 
  float xSlSL3[MAX_SEGMENT]; 
  float dxdzSlSL3[MAX_SEGMENT]; 
  float exSlSL3[MAX_SEGMENT]; 
  float edxdzSlSL3[MAX_SEGMENT]; 
  float exdxdzSlSL3[MAX_SEGMENT]; 
  float xSL3SL1[MAX_SEGMENT]; 
  float xc[MAX_SEGMENT][MAX_HIT_CHAM];
  float yc[MAX_SEGMENT][MAX_HIT_CHAM];
  float zc[MAX_SEGMENT][MAX_HIT_CHAM];
  float ex[MAX_SEGMENT][MAX_HIT_CHAM];
  float xcp[MAX_SEGMENT][MAX_HIT_CHAM];
  float ycp[MAX_SEGMENT][MAX_HIT_CHAM];
  float excp[MAX_SEGMENT][MAX_HIT_CHAM];
  float eycp[MAX_SEGMENT][MAX_HIT_CHAM];
  int wh[MAX_SEGMENT]; int st[MAX_SEGMENT]; int sr[MAX_SEGMENT];
  int sl[MAX_SEGMENT][MAX_HIT_CHAM];
  int la[MAX_SEGMENT][MAX_HIT_CHAM];
  //---------------------------------------------------------------

  
  

  // private data members
  TFile *f;
  TTree *ttreeOutput;
  TChain *tali;


  AlignmentParameterStore* theAlignmentParameterStore;
  std::vector<Alignable*> theAlignables;
  AlignableNavigator* theAlignableDetAccessor;
  
  //Service for histograms
  edm::Service<TFileService> fs;

  edm::InputTag globalTracks;
  edm::InputTag consTraj;
  std::string ntuplePath;
  float ptMax;
  float ptMin;
  float nPhihits;
  float nThetahits;
  int workingmode;
  int numberOfRootFiles;
  int nMtxSection;


  //FIXME: Not clear if needed
  float numberOfSigmasX;
  float numberOfSigmasDXDZ;
  float numberOfSigmasY;
  float numberOfSigmasDYDZ;

  float meanx[5][4][14];
  float sigmax[5][4][14];
  float meandxdz[5][4][14];
  float sigmadxdz[5][4][14];
  float meany[5][4][14];
  float sigmay[5][4][14];
  float meandydz[5][4][14];
  float sigmadydz[5][4][14];


};

#endif
