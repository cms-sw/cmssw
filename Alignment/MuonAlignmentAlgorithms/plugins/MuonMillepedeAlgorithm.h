#ifndef Alignment_MuonStandaloneAlgorithm_MuonMillepedeAlgorithm_h
#define Alignment_MuonStandaloneAlgorithm_MuonMillepedeAlgorithm_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"


#include "TH1D.h"
#include "TTree.h"
#include "TFile.h"
#include "TMatrixD.h"

class AlignableNavigator;
class TFile;
class TTree;

class MuonMillepedeAlgorithm : public AlignmentAlgorithmBase
{

 public:
  
  /// Constructor
  MuonMillepedeAlgorithm(const edm::ParameterSet& cfg);

  /// Destructor
  ~MuonMillepedeAlgorithm() {};

  /// Call at beginning of job
  void initialize( const edm::EventSetup& setup, 
                   AlignableTracker* tracker, AlignableMuon* muon, 
                   AlignmentParameterStore* store);

  /// Call at end of job
  void terminate(void);



  /// Run the algorithm on trajectories and tracks
  void run( const edm::EventSetup& setup, const ConstTrajTrackPairCollection& tracks );

  void updateInfo(AlgebraicMatrix, AlgebraicMatrix, AlgebraicMatrix, std::string);
 
  void toTMat(AlgebraicMatrix *, TMatrixD *);

  void collect();
 
 private:

  // private data members
 
  void printM(AlgebraicMatrix ); 
  
  AlignmentParameterStore* theAlignmentParameterStore;
  std::vector<Alignable*> theAlignables;
  AlignableNavigator* theAlignableDetAccessor;

  // verbosity flag
  bool verbose;

  //Store residuals
  std::map<std::string, TH1D *> histoMap;
  
  std::map<std::string, AlgebraicMatrix *> map_invCov;
  std::map<std::string, AlgebraicMatrix *> map_weightRes;
  std::map<std::string, AlgebraicMatrix *> map_N;

  double ptCut, chi2nCut;


  //Service for histograms
  edm::Service<TFileService> fs;


  std::string collec_f;
  std::string outputCollName;
  bool isCollectionJob;
  std::string collec_path; 
  int collec_number;

  

};

#endif
