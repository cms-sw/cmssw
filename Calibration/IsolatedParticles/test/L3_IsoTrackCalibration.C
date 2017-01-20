//////////////////////////////////////////////////////////
// L3 iterative procedure 
// for IsoTrack calibration
//
// CalibTree class contains ROOT-tree
// generated with IsoTrackCalibration plugin
//
// version 5.0   September 2016
// modified detID for CMSSW_8_X !!!!!!!!!!!!!!!!!!!!!
// M. Chadeeva
//////////////////////////////////////////////////////////

#include <TSystem.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
//#include <TProfile.h>
#include <TLegend.h>
#include <TString.h>
#include <TF1.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>

//**********************************************************
// Constants
//**********************************************************
const unsigned int MAXNUM_SUBDET = 100;
const int FIRST_IETA_TR = 15;
const int FIRST_IETA_HE = 18;
const int FIRST_IETA_FWD_1 = 25;
const int FIRST_IETA_FWD_2 = 26;
const unsigned int N_DEPTHS = 3;
/*
// old detID format for CMSSW_7_X 
const unsigned int PHI_MASK       = 0x7F;
const unsigned int ETA_OFFSET     = 7;
const unsigned int ETA_MASK       = 0x3F;
const unsigned int ZSIDE_MASK     = 0x2000;
const unsigned int DEPTH_OFFSET   = 14;
const unsigned int DEPTH_MASK     = 0x1F;
const unsigned int DEPTH_SET      = 0x1C000;
unsigned int MASK(0xFF80); //merge phi 
*/
// new detID format for CMSSW_8_X
const unsigned int PHI_MASK       = 0x3FF;
const unsigned int ETA_OFFSET     = 10;
const unsigned int ETA_MASK       = 0x1FF;
const unsigned int ZSIDE_MASK     = 0x80000;
const unsigned int DEPTH_OFFSET   = 20;
const unsigned int DEPTH_MASK     = 0xF;
const unsigned int DEPTH_SET      = 0xF00000;

//--------------------------------------------
// merging depths

const unsigned int MERGE_PHI_AND_DEPTHS = 1;
const unsigned int MASK(0xFFC00); //merge phi and depth
/*
const unsigned int MERGE_PHI_AND_DEPTHS = 0;
const unsigned int MASK(0xFFFC00); //merge phi
*/
//-------------------------------------------

// individual ieta rings
const unsigned int MASK2(0); // no second mask
const int N_ETA_RINGS_PER_BIN = 1;
/*
// twin (even+odd) ieta rings
const unsigned int MASK2(0x80);
const int N_ETA_RINGS_PER_BIN = 2;

// 4-fold ieta rings
const unsigned int MASK2(0x180);
const int N_ETA_RINGS_PER_BIN = 4;
*/
const int MAX_ONESIDE_ETA_RINGS = 30;
const int HALF_NUM_ETA_BINS =
  (MAX_ONESIDE_ETA_RINGS + 1*(N_ETA_RINGS_PER_BIN>1))/N_ETA_RINGS_PER_BIN;
const int NUM_ETA_BINS = 2*HALF_NUM_ETA_BINS + 1;

const int MIN_N_TRACKS_PER_CELL = 50;
const int MIN_N_ENTRIES_FOR_FIT = 150;
const double MAX_REL_UNC_FACTOR = 0.2;

const bool APPLY_CORRECTION_FOR_PU = true;
const bool SINGLE_REFERENCE_RESPONSE = false;
/*
//--- base correction for PU as of 2015 MC studies
const char *l3prefix1 = "base";
const double DELTA_CUT = 0.0; 
const double LINEAR_COR_COEF[5] = { -0.375, -0.375, -0.375, -0.375, -0.375 };
const double SQUARE_COR_COEF[5] = { -0.450, -0.450, -0.450, -0.450, -0.450 };
//const double UPPER_LIMIT_DELTA_PU_COR = 2.0;
*/
//--- optimized correction for PU as of 2016 MC studies
// tuned to get MPV after correction
// at the level of that of single pion response w/o PU and w/o correction
const char *l3prefix1 = "opt";
const double DELTA_CUT = 0.02; 
const double LINEAR_COR_COEF[5] = { -0.35, -0.35, -0.35, -0.35, -0.45 };
const double SQUARE_COR_COEF[5] = { -0.65, -0.65, -0.65, -0.30, -0.10 };
//const double UPPER_LIMIT_DELTA_PU_COR = 1.5;

const double UPPER_LIMIT_RESPONSE_BEFORE_COR = 3.0;
const double FLEX_SEL_FIRST_CONST = 20.0;  // 20.(for exp); or 16*2
const double FLEX_SEL_SECOND_CONST = 18.0; // 18.(for exp);

const double MIN_RESPONSE_HIST = 0.0;
const double MAX_RESPONSE_HIST = UPPER_LIMIT_RESPONSE_BEFORE_COR;
const int NBIN_RESPONSE_HIST = 480;
const int NBIN_RESPONSE_HIST_IND = 120;
const double FIT_RMS_INTERVAL = 1.5;
const double RESOLUTION_HCAL = 0.3;
const double LOW_RESPONSE = 0.5;
const double HIGH_RESPONSE = 1.5;

//std::cout.precision(3);

//**********************************************************
// Header with CalibTree class definition
//**********************************************************

//////////////////////////////////////////////////////////
// Header with CalibTree class
// for L3 iterative procedure
// implemented in L3_IsoTrackCalibration.C
// 
// version 2.0 January 2016
// M. Chadeeva
//////////////////////////////////////////////////////////

#include <TSystem.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TProfile.h>
#include <TLegend.h>
#include <TString.h>
#include <TF1.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>


//**********************************************************
// Class with TTree containing parameters of selected events
//**********************************************************
class CalibTree {
public :
  TChain          *fChain;   //!pointer to the analyzed TTree
  //TChain          *inChain;   //!pointer to the analyzed TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t           t_Run;
  Int_t           t_Event;
  Int_t           t_nVtx;
  Int_t           t_nTrk;
  Double_t        t_EventWeight;
  Double_t        t_p;
  Double_t        t_pt;
  Int_t           t_ieta;
  Double_t        t_phi;
  Double_t        t_eMipDR;
  Double_t        t_eHcal;
  Double_t        t_eHcal10;
  Double_t        t_eHcal30;
  Double_t        t_hmaxNearP;
  Bool_t          t_selectTk;
  Bool_t          t_qltyMissFlag;
  Bool_t          t_qltyPVFlag;
/*
  Double_t        t_l1pt;
  Double_t        t_l1eta;
  Double_t        t_l1phi;
  Double_t        t_l3pt;
  Double_t        t_l3eta;
  Double_t        t_l3phi;
*/
  Double_t        t_mindR1;
  Double_t        t_mindR2;
  std::vector<unsigned int> *t_DetIds;
  //std::vector<unsigned int> *t_DetIds1;
  //std::vector<unsigned int> *t_DetIds3;
  std::vector<double>  *t_HitEnergies;
  //std::vector<double>  *t_HitEnergies1;
  //std::vector<double>  *t_HitEnergies3;
  
  // List of branches
  TBranch        *b_t_Run;   //!
  TBranch        *b_t_Event;   //!
  TBranch        *b_t_nVtx;
  TBranch        *b_t_nTrk;
  TBranch        *b_t_EventWeight;   //!
  TBranch        *b_t_p;   //!
  TBranch        *b_t_pt;   //!
  TBranch        *b_t_ieta;   //!
  TBranch        *b_t_phi;   //!
  TBranch        *b_t_eMipDR;   //!
  TBranch        *b_t_eHcal;   //!
  TBranch        *b_t_eHcal10;   //!
  TBranch        *b_t_eHcal30;   //!
  TBranch        *b_t_hmaxNearP;   //!
  TBranch        *b_t_selectTk;   //!
  TBranch        *b_t_qltyMissFlag;   //!
  TBranch        *b_t_qltyPVFlag;   //!
/*
  TBranch        *b_t_l1pt;   //!
  TBranch        *b_t_l1eta;   //!
  TBranch        *b_t_l1phi;   //!
  TBranch        *b_t_l3pt;   //!
  TBranch        *b_t_l3eta;   //!
  TBranch        *b_t_l3phi;   //!
*/
  TBranch        *b_t_mindR1;   //!
  TBranch        *b_t_mindR2;   //!
  TBranch        *b_t_DetIds;   //!
  //TBranch        *b_t_DetIds1;   //!
  //TBranch        *b_t_DetIds3;   //!
  TBranch        *b_t_HitEnergies;   //!
  //TBranch        *b_t_HitEnergies1;   //!
  //TBranch        *b_t_HitEnergies3;   //!

  //--- constructor & destructor
  //CalibTree(TTree *tree=0);
  CalibTree(TChain *tree,
	    double min_enrHcal, double min_pt,
	    double lim_mipEcal, double lim_charIso,
	    double min_trackMom, double max_trackMom);
  virtual ~CalibTree();
  
  //--- functions
  virtual Int_t      GetEntry(Long64_t entry);
  virtual Long64_t   LoadTree(Long64_t entry);
  //virtual void     Init(TTree *tree);
  virtual void       Init(TChain *tree);
  virtual Bool_t     Notify();
  virtual Int_t      firstLoop(unsigned, bool, unsigned);
  virtual Double_t   loopForIteration(unsigned, unsigned, unsigned);
  virtual Double_t   lastLoop(unsigned, unsigned, bool, unsigned);
  Bool_t             goodTrack(int);
  Bool_t             getFactorsFromFile(std::string, unsigned);
  unsigned int       saveFactorsInFile(std::string);
  Bool_t             openOutputRootFile(std::string);

  //--- variables for iterations
  Double_t referenceResponse;
  Double_t referenceResponseHB;
  Double_t referenceResponseTR;
  Double_t referenceResponseHE;
  Double_t maxZtestFromWeights;
  Double_t maxSys2StatRatio;
  int maxNumOfTracksForIeta;
  std::map<unsigned int, double> factors;
  std::map<unsigned int, double> uncFromWeights;
  std::map<unsigned int, double> uncFromDeviation;
  std::map<unsigned int, int> subDetector_trk;
  std::map<unsigned int, int> subDetector_final;
  std::map<unsigned int, int> nTrks;
  std::map<unsigned int, int> nSubdetInEvent;
  std::map<unsigned int, int> nPhiMergedInEvent;
  std::map<unsigned int, double> sumOfResponse;
  std::map<unsigned int, double> sumOfResponseSquared;

  //--- variables for selection
  double minEnrHcal;
  double minTrackPt;
  double minTrackMom;
  double maxTrackMom;
  double limMipEcal;
  double limCharIso;
  double constForFlexSel;
  
  //--- variables for plotting
  TFile *foutRootFile;
  TH1F* e2p_init;
  TH1F* e2pHB_init;
  TH1F* e2pTR_init;
  TH1F* e2pHE_init;
  TH1F* e2p_last;
  TH1F* e2pHB_last;
  TH1F* e2pTR_last;
  TH1F* e2pHE_last;
  
  TH1F* ieta_lefttail;
  TH1F* ieta_righttail;
  /*
  TProfile* deltaVSieta;
  TProfile* deltaNorm;
  TProfile* eHcalDeltaHB;
  TProfile* eHcalDeltaTR;
  TProfile* eHcalDeltaHE;
  TProfile* eHcalDeltaHEfwd;
  TProfile* frespDeltaHB;
  TProfile* frespDeltaTR;
  TProfile* frespDeltaHE;
  TProfile* frespDeltaHEfwd;
  */
};


//**********************************************************
// Description of function to run iteration
//**********************************************************
unsigned int runIterations(const char *inFileDir = ".", 
			   const char *inFileNamePrefix = "outputFromAnalyzer",
			   const int firstInputFileEnum = 0,
			   const int lastInputFileEnum = 1,
			   const unsigned maxNumberOfIterations = 10,
			   //const char *l3prefix0 = "cf",
			   const double minHcalEnergy = 10.0,
			   const double minPt = 7.0,
			   const double limitForChargeIsolation = 2.0,
			   const double minTrackMomentum = 40.0,
			   const double maxTrackMomentum = 60.0,
			   const double limitForMipInEcal = 1.0,
			   const bool shiftResponse = 1,
			   const unsigned int subSample = 2,
			   const bool isCrosscheck = false,
			   const char *inTxtFilePrefix = "test",
			   const char *treeDirName = "IsoTrackCalibration", 
			   const char *treeName = "CalibTree",
			   unsigned int Debug = 0)
{
  // Debug:  0-no debugging; 1-short debug; >1 - number of events to be shown in detail
  // subSample: extract factors from odd (0), even(1) or all(2) events
  // limitForChargeIsolation:  <0 - flex. sel.
  // and corr. for PU
  
  if ( isCrosscheck )
    std::cout << "Test with previously extracted factors..." << std::endl;
  else
    std::cout << "Extracting factors using L3 algorithm and isolated tracks..." << std::endl;

  char l3prefix0[10] = "ref1";
  if ( !SINGLE_REFERENCE_RESPONSE ) sprintf(l3prefix0,"ref3");
  char l3prefix2[20] = "_noCor";
  if ( APPLY_CORRECTION_FOR_PU ) sprintf(l3prefix2,"_%s%02d",
					 l3prefix1, int(100*DELTA_CUT)
					 );				
  char l3prefix3[10] = "_mean";
  if ( shiftResponse ) sprintf(l3prefix3,"_mpv");
  char l3prefix4[8] = "_3dep";
  if ( MERGE_PHI_AND_DEPTHS ) sprintf(l3prefix4,"_merged");
  char l3prefix[40];
  sprintf(l3prefix,"%s%s%s%s", l3prefix0, l3prefix2, l3prefix3, l3prefix4);

  char isoPrefix[10] = "hard";
  if ( limitForChargeIsolation < 0 ) sprintf(isoPrefix, "flex");
    
  char fnameInput[120];
  char fnameOutRoot[120];
  char fnameOutTxt[120] = "dummy";
  char fnameInTxt[120]  = "dummy";
  char tname[100];
  
  TGraph *g_converge1 = new TGraph(maxNumberOfIterations);
  TGraph *g_converge2 = new TGraph(maxNumberOfIterations);
  TGraph *g_converge3 = new TGraph(maxNumberOfIterations);

  sprintf(tname, "%s/%s", treeDirName, treeName );
  TChain tree(tname);

  //--- combine tree from several enumerated files with the same prefix
  //    or one file w/o number (firstInputFileEnum = lastInputFileEnum < 0 )

  for ( int ik = firstInputFileEnum; ik <= lastInputFileEnum; ik++ ) {
    if ( ik < 0 ) 
      sprintf(fnameInput, "%s/%s.root", inFileDir, inFileNamePrefix);
    else if (ik < 10 )
      sprintf(fnameInput, "%s/%s_%1d.root", inFileDir, inFileNamePrefix, ik);
    else if (ik < 100 )
      sprintf(fnameInput, "%s/%s_%2d.root", inFileDir, inFileNamePrefix, ik);
    else if (ik < 1000 )
      sprintf(fnameInput, "%s/%s_%3d.root", inFileDir, inFileNamePrefix, ik);
    else
      sprintf(fnameInput, "%s/%s_%4d.root", inFileDir, inFileNamePrefix, ik);

    if ( !gSystem->Which("./", fnameInput ) ) { // check file availability
      std::cout << "File " << fnameInput << " doesn't exist." << std::endl;
    }
    else {
      tree.Add(fnameInput);
      std::cout << "Add tree from " << fnameInput 
	        << "   total number of entries (tracks): "
		<< tree.GetEntries() << std::endl;
    }
  }
  if ( tree.GetEntries() == 0 ) {
    std:: cout << "Tree is empty." << std::endl;
    return -2;
  }

  //--- Initialize tree
  CalibTree t(&tree,
	      minHcalEnergy, minPt,
	      limitForMipInEcal, limitForChargeIsolation,
	      minTrackMomentum, maxTrackMomentum);
  
  //--- Define files
  if ( isCrosscheck ) {
    sprintf(fnameInTxt, "%s_%s%02d-%02d-%02d_p%02d-%02d_pt%02d_eh%02d_ee%1d_step%1d.txt",
	    inTxtFilePrefix,
	    isoPrefix, int(t.limCharIso),
	    int(abs(FLEX_SEL_FIRST_CONST)),
	    int(abs(FLEX_SEL_SECOND_CONST)),
	    int(minTrackMomentum), int(maxTrackMomentum), int(minPt),
	    int(minHcalEnergy), int(limitForMipInEcal),
	    N_ETA_RINGS_PER_BIN
	    );
    sprintf(fnameOutRoot,
	    "test_%1d_%s_by_%s_%s%02d-%02d-%02d_p%02d-%02d_pt%02d_eh%02d_ee%1d_step%1d.root",
	    subSample,
	    inFileNamePrefix,
	    inTxtFilePrefix,
	    isoPrefix, int(t.limCharIso),
	    int(abs(FLEX_SEL_FIRST_CONST)),
	    int(abs(FLEX_SEL_SECOND_CONST)),
	    int(minTrackMomentum), int(maxTrackMomentum), int(minPt),
	    int(minHcalEnergy), int(limitForMipInEcal),
	    N_ETA_RINGS_PER_BIN
	    );
  }
  else {    
    sprintf(fnameOutTxt,
	    "%s_%1d_%s_i%02d_%s%02d-%02d-%02d_p%02d-%02d_pt%02d_eh%02d_ee%1d_step%1d.txt",
	    l3prefix,
	    subSample,
	    inFileNamePrefix,
	    maxNumberOfIterations,
	    isoPrefix, int(t.limCharIso),
	    int(abs(FLEX_SEL_FIRST_CONST)),
	    int(abs(FLEX_SEL_SECOND_CONST)),
	    int(minTrackMomentum), int(maxTrackMomentum), int(minPt),
	    int(minHcalEnergy), int(limitForMipInEcal),
	    N_ETA_RINGS_PER_BIN
	    );
    sprintf(fnameOutRoot,
	    "%s_%1d_%s_i%02d_%s%02d-%02d-%02d_p%02d-%02d_pt%02d_eh%02d_ee%1d_step%1d.root",
	    l3prefix,
	    subSample,
	    inFileNamePrefix,
	    maxNumberOfIterations,
	    isoPrefix, int(t.limCharIso),
	    int(abs(FLEX_SEL_FIRST_CONST)),
	    int(abs(FLEX_SEL_SECOND_CONST)),
	    int(minTrackMomentum), int(maxTrackMomentum), int(minPt),
	    int(minHcalEnergy), int(limitForMipInEcal),
	    N_ETA_RINGS_PER_BIN
	    );
  }  
  if ( !t.openOutputRootFile(fnameOutRoot) ) {
    std::cout << "Problems with booking output file " << fnameOutRoot << std::endl;
    return -1;
  }
  std::cout << "Correction for PU: ";
  if ( APPLY_CORRECTION_FOR_PU ) {
    std::cout << " applied for delta > " << DELTA_CUT << std::endl;
    std::cout << " for HB: " << LINEAR_COR_COEF[0]
	      << " ; " << SQUARE_COR_COEF[0]
	      << std::endl;
    std::cout << " for TR: " << LINEAR_COR_COEF[1]
	      << " ; " << SQUARE_COR_COEF[1]
	      << std::endl;
    std::cout << " for HE(<=24): " << LINEAR_COR_COEF[2]
	      << " ; " << SQUARE_COR_COEF[2]
	      << std::endl;
    std::cout << " for ieta=25: " << LINEAR_COR_COEF[3]
	      << " ; " << SQUARE_COR_COEF[3]
	      << std::endl;
    std::cout << " for ieta=26: " << LINEAR_COR_COEF[4]
	      << " ; " << SQUARE_COR_COEF[4]
	      << std::endl;
  }
  else
    std::cout << " no " << std::endl;
    
  /*
  std::cout << "Constant coefficient from charge isolation: "
	    << t.constForFlexSel << std::endl; 
  */

  unsigned int numOfSavedFactors(0);
  int nEventsWithGoodTrack(0);
  double MPVfromLastFit(0);
  
  if ( isCrosscheck ) {
    // open txt file and fill map with factors
    if ( t.getFactorsFromFile(fnameInTxt, Debug) ) {
      nEventsWithGoodTrack = t.firstLoop(subSample, false, Debug);
      std::cout << "Number of events with good track = "
		<< nEventsWithGoodTrack << std::endl;
      MPVfromLastFit = t.lastLoop(subSample, maxNumberOfIterations, true, Debug);
      std::cout << "Finish testing " << t.factors.size() << " factors from file "
		<< fnameInTxt << std::endl;
      std::cout << "MPV from fit after last iteration = "
		<< MPVfromLastFit << std::endl;
      std::cout << "Test plots saved in " << fnameOutRoot << std::endl;
    }
    else {
      std::cout << "File " << fnameInTxt << " doesn't exist." << std::endl;
    }
  }
  else {
    //--- Prepare initial histograms and count good track
    nEventsWithGoodTrack = t.firstLoop(subSample, shiftResponse, Debug);
    std::cout << "Number of events with good track = "
	      << nEventsWithGoodTrack << std::endl;
    //--- Iterate
    for ( unsigned int k = 0; k < maxNumberOfIterations; ++k ) {
      g_converge1->SetPoint( k, k+1, t.loopForIteration(subSample, k+1, Debug) );
      g_converge2->SetPoint( k, k+1, t.maxZtestFromWeights );
      g_converge3->SetPoint( k, k+1, t.maxSys2StatRatio );
    }
    //--- Finish
    MPVfromLastFit = t.lastLoop(subSample, maxNumberOfIterations, false, Debug);
    numOfSavedFactors = t.saveFactorsInFile(fnameOutTxt);

    sprintf(tname,"Mean deviation for subdetectors with Ntrack>%d",
	    MIN_N_TRACKS_PER_CELL);
    g_converge1->SetTitle(tname);
    g_converge1->GetXaxis()->SetTitle("iteration");
    t.foutRootFile->WriteTObject(g_converge1, "g_cvgD");
    sprintf(tname,"Max abs(Z-test) for factors");
    g_converge2->SetTitle(tname);
    g_converge2->GetXaxis()->SetTitle("iteration");
    t.foutRootFile->WriteTObject(g_converge2, "g_cvgW");
    sprintf(tname,"Max ratio of syst. to stat. uncertainty");
    g_converge3->SetTitle(tname);
    g_converge3->GetXaxis()->SetTitle("iteration");
    t.foutRootFile->WriteTObject(g_converge3, "g_cvgR");
    
    std::cout << "Finish adjusting factors after "
	      << maxNumberOfIterations << " iterations" << std::endl;
    std::cout << "MPV from fit after last iteration = "
	      << MPVfromLastFit << std::endl;
    std::cout << "Table with " << numOfSavedFactors << " factors"
	      << " with more than " << MIN_N_TRACKS_PER_CELL << " tracks/subdetector"
	      << " (from " << t.factors.size() << " available)"
	      << " is written in file " << fnameOutTxt << std::endl;
    std::cout << "Plots saved in " << fnameOutRoot << std::endl;
  }

  return numOfSavedFactors;
}

//**********************************************************
// CalibTree constructor
//**********************************************************
//CalibTree::CalibTree(TTree *tree) : fChain(0) {
CalibTree::CalibTree(TChain *tree,
		     double min_enrHcal,
		     double min_pt,
		     double lim_mipEcal,
		     double lim_charIso,
		     double min_trackMom,
		     double max_trackMom )
{ //: fChain(0) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  if (tree == 0) {
    TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("output.root");
    if (!f || !f->IsOpen()) {
      f = new TFile("output.root");
    }
    TDirectory * dir = (TDirectory*)f->Get("IsoTrackCalibration");
    dir->GetObject("CalibTree",tree);
  }
  Init(tree);

  referenceResponse = 1;
  maxNumOfTracksForIeta = 0;
  // initialization of maps
  factors.clear();
  uncFromWeights.clear();
  uncFromDeviation.clear();
  subDetector_trk.clear();
  subDetector_final.clear();
  nTrks.clear();
  nSubdetInEvent.clear();
  nPhiMergedInEvent.clear();
  sumOfResponse.clear();
  sumOfResponseSquared.clear();
  
  // selection
  minEnrHcal = min_enrHcal;
  minTrackPt = min_pt;
  minTrackMom = min_trackMom;
  maxTrackMom = max_trackMom;
  limMipEcal = lim_mipEcal;
  limCharIso = abs(lim_charIso);
  if ( lim_charIso < 0 ) 
    constForFlexSel = log(FLEX_SEL_FIRST_CONST/limCharIso)/FLEX_SEL_SECOND_CONST;
  else constForFlexSel = 0;
}

//**********************************************************
// CalibTree destructor
//**********************************************************
CalibTree::~CalibTree() {

    foutRootFile->cd();
    foutRootFile->Write();
    foutRootFile->Close();

  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

//**********************************************************
// Get entry function
//**********************************************************
Int_t CalibTree::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

//**********************************************************
// Load tree function
//**********************************************************
Long64_t CalibTree::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain) return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0) return centry;
  if (fChain->GetTreeNumber() != fCurrent) {
    fCurrent = fChain->GetTreeNumber();
    Notify();
  }
  return centry;
}

//**********************************************************
// Initialisation of TTree
//**********************************************************
void CalibTree::Init(TChain *tree) {
  // Set object pointer
  t_DetIds = 0;
  //t_DetIds1 = 0;
  //t_DetIds3 = 0;
  t_HitEnergies = 0;
  //t_HitEnergies1 = 0;
  //t_HitEnergies3 = 0;
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_nVtx", &t_nVtx, &b_t_nVtx);
  fChain->SetBranchAddress("t_nTrk", &t_nTrk, &b_t_nTrk);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_pt", &t_pt, &b_t_pt);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_phi", &t_phi, &b_t_phi);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_eHcal10", &t_eHcal10, &b_t_eHcal10);
  fChain->SetBranchAddress("t_eHcal30", &t_eHcal30, &b_t_eHcal30);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
/*
  fChain->SetBranchAddress("t_l1pt", &t_l1pt, &b_t_l1pt);
  fChain->SetBranchAddress("t_l1eta", &t_l1eta, &b_t_l1eta);
  fChain->SetBranchAddress("t_l1phi", &t_l1phi, &b_t_l1phi);
  fChain->SetBranchAddress("t_l3pt", &t_l3pt, &b_t_l3pt);
  fChain->SetBranchAddress("t_l3eta", &t_l3eta, &b_t_l3eta);
  fChain->SetBranchAddress("t_l3phi", &t_l3phi, &b_t_l3phi);
*/
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  //fChain->SetBranchAddress("t_DetIds1", &t_DetIds1, &b_t_DetIds1);
  //fChain->SetBranchAddress("t_DetIds3", &t_DetIds3, &b_t_DetIds3);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  //fChain->SetBranchAddress("t_HitEnergies1", &t_HitEnergies1, &b_t_HitEnergies1);
  //fChain->SetBranchAddress("t_HitEnergies3", &t_HitEnergies3, &b_t_HitEnergies3);
  Notify();
}

//**********************************************************
// Notification when opening new file
//**********************************************************
Bool_t CalibTree::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}
//**********************************************************
// Open file and book histograms
//**********************************************************
bool CalibTree::openOutputRootFile(std::string fname)
{
  bool decision = false;
  
  foutRootFile = new TFile(fname.c_str(), "RECREATE");
  if ( foutRootFile != NULL ) decision = true;  
  foutRootFile->cd();

  return decision;
}

//**********************************************************
// Initial loop over events in the tree
//**********************************************************
Int_t CalibTree::firstLoop(unsigned int subsample,
			   bool shiftResp,
			   unsigned int debug)
{
  char name[100];
  unsigned int ndebug(0);
  double maxRespForGoodTrack(0);
  double minRespForGoodTrack(1000);
  int nRespOverHistLimit(0);
  int ntrk_ieta[NUM_ETA_BINS];
  for ( int j = 0; j < NUM_ETA_BINS; j++ ) {
    ntrk_ieta[j] = 0;
  }
  
  char scorr[80] = "correction for PU";
  char sxlabel[80] ="(E^{cor}_{hcal} + E_{ecal})/p_{track}"; 
  if ( !APPLY_CORRECTION_FOR_PU ) {
    sprintf(scorr,"no correction for PU");
    sprintf(sxlabel,"(E_{hcal} + E_{ecal})/p_{track}");
  }
  
  TF1* f1 = new TF1("f1","gaus", MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);

  //--------- initialize histograms for response -----------------------------------------
  sprintf(name,"Initial HB+HE: %s", scorr);
  e2p_init = new TH1F("e2p_init", name,
		      NBIN_RESPONSE_HIST, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2p_init->Sumw2();
  e2p_init->GetXaxis()->SetTitle(sxlabel);
  
  sprintf(name,"Initial HB: %s", scorr);
  e2pHB_init = new TH1F("e2pHB_init", name,
			NBIN_RESPONSE_HIST/2, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2pHB_init->Sumw2();
  e2pHB_init->GetXaxis()->SetTitle(sxlabel);

  sprintf(name,"Initial TR: %s", scorr);
  e2pTR_init = new TH1F("e2pTR_init", name,
			NBIN_RESPONSE_HIST/10, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2pTR_init->Sumw2();
  e2pTR_init->GetXaxis()->SetTitle(sxlabel);

  sprintf(name,"Initial HE: %s", scorr);
  e2pHE_init = new TH1F("e2pHE_init", name,
			NBIN_RESPONSE_HIST/2, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2pHE_init->Sumw2();
  e2pHE_init->GetXaxis()->SetTitle(sxlabel);

  sprintf(name,"Response < %3.1f", LOW_RESPONSE);
  ieta_lefttail = new TH1F("ieta_lefttail", name,
			   2*MAX_ONESIDE_ETA_RINGS,
			   -MAX_ONESIDE_ETA_RINGS, MAX_ONESIDE_ETA_RINGS); 
  ieta_lefttail->GetXaxis()->SetTitle("i#eta");

  sprintf(name,"Response > %3.1f", HIGH_RESPONSE);
  ieta_righttail = new TH1F("ieta_righttail", name,
			   2*MAX_ONESIDE_ETA_RINGS,
			   -MAX_ONESIDE_ETA_RINGS, MAX_ONESIDE_ETA_RINGS); 			    
  ieta_righttail->GetXaxis()->SetTitle("i#eta");
  
//--- initialize chain ----------------------------------------
  if (fChain == 0) return 0;
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nb = 0;
  
  int nSelectedEvents(0);

  if ( debug > 0 ) { 
    std::cout << "---------- First loop -------------------------- " << std::endl;
  }
// ----------------------- loop over events -------------------------------------  
  for (Long64_t jentry=0; jentry<nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if ( ientry < 0 || ndebug > debug ) break;   
    nb = fChain->GetEntry(jentry);   //nbytes += nb;
    
    if ( (jentry%2 == subsample) ) continue;   // only odd or even events
 
// --------------- selection of good track --------------------    

    if ( !goodTrack(t_ieta) ) continue;
    
    if ( debug > 1 ) {
      ndebug++;
      std::cout << "***Entry (Track) Number : " << ientry << "(" << jentry << ")"
		<< " p/eHCal/eMipDR/nDets : " << t_p << "/" << t_eHcal 
		<< "/" << t_eMipDR << "/" << (*t_DetIds).size() 
		<< std::endl;
    }
    
    double eTotal(0.0);
    double eTotalWithEcal(0.0);
      
    // ---- loop over active subdetectors in the event for total energy ---
    unsigned int nDets = (*t_DetIds).size();
    for (unsigned int idet = 0; idet < nDets; idet++) { 
      eTotal += (*t_HitEnergies)[idet];
    }
    eTotalWithEcal = eTotal + t_eMipDR;    

// --- Correction for PU  --------
    double eTotalCor(eTotal);
    double eTotalWithEcalCor(eTotalWithEcal);
    //double e10(0.0);
    //double e30(0.0);
    double correctionForPU(1.0);
    double de2p(0.0);
    int abs_t_ieta = abs(t_ieta);
    
    if ( APPLY_CORRECTION_FOR_PU ) { 
      /*
      for (unsigned int idet1 = 0; idet1 < (*t_DetIds1).size(); idet1++) { 
	e10 += (*t_HitEnergies1)[idet1];
      }
      for (unsigned int idet3 = 0; idet3 < (*t_DetIds3).size(); idet3++) { 
	e30 += (*t_HitEnergies3)[idet3];
      }
      
      de2p = (e30 - e10)/t_p;
      */
      de2p = (t_eHcal30 - t_eHcal10)/t_p;

      if ( de2p > DELTA_CUT ) {
	int icor = int(abs_t_ieta >= FIRST_IETA_TR) + int(abs_t_ieta >= FIRST_IETA_HE)
	  + int(abs_t_ieta >= FIRST_IETA_FWD_1) + int(abs_t_ieta >= FIRST_IETA_FWD_2);
	correctionForPU = (1 + LINEAR_COR_COEF[icor]*(t_eHcal/t_p)*de2p
			   *(1 + SQUARE_COR_COEF[icor]*de2p));
      }
    }    

    // check for possibility to correct for PU
    if ( correctionForPU <= 0 || correctionForPU > 1 ) continue;
    nSelectedEvents++;

    eTotalCor = eTotal*correctionForPU;
    eTotalWithEcalCor = eTotalCor + t_eMipDR;

    double response = eTotalWithEcalCor/t_p;
    std::map<unsigned int, bool> sameSubdet;
    sameSubdet.clear();
    double resp2 = response*response;

    for (unsigned int idet = 0; idet < nDets; idet++) { 
      unsigned int detId = ( (*t_DetIds)[idet] & MASK ) | MASK2 ;

      if ( debug > 1 ) {
	unsigned int detId0 = ( (*t_DetIds)[idet] & MASK ) ;
	std::cout << "jentry/idet/detId :: ieta/z/depth ::: "
		  << std::dec
		  << jentry << " / "
		  << ((*t_DetIds)[idet]) << " / "
		  << detId0 << "(" << detId << ")" << " :: "
		  << ((detId0>>ETA_OFFSET) & ETA_MASK)
		  << "(" << ((detId>>ETA_OFFSET) & ETA_MASK) << ")" << " / "
		  << ((detId0&ZSIDE_MASK) ? 1 : -1)
		  << "(" << ((detId&ZSIDE_MASK) ? 1 : -1) << ")" << " / "
		  << ((detId0>>DEPTH_OFFSET)&DEPTH_MASK)
		  << "(" << ((detId>>DEPTH_OFFSET)&DEPTH_MASK) << ")"
		  << std::endl;
      }
      if (nPhiMergedInEvent.find(detId) != nPhiMergedInEvent.end()) 
	nPhiMergedInEvent[detId]++;
      else 
	nPhiMergedInEvent.insert(std::pair<unsigned int,int>(detId, 1));
		
      if (nTrks.find(detId) != nTrks.end()) {
	if ( sameSubdet.find(detId) == sameSubdet.end() ) {
	  nTrks[detId]++;
	  nSubdetInEvent[detId] += nDets;
	  sumOfResponse[detId] += response;
	  sumOfResponseSquared[detId] += resp2;
	  sameSubdet.insert(std::pair<unsigned int,bool>(detId, true));
	}
      }
      else {
	nTrks.insert(std::pair<unsigned int,int>(detId, 1));
	nSubdetInEvent.insert(std::pair<unsigned int,int>(detId, nDets));
	sumOfResponse.insert(std::pair<unsigned int,double>(detId,response));
	sumOfResponseSquared.insert(std::pair<unsigned int,double>(detId,resp2));
	sameSubdet.insert(std::pair<unsigned int,bool>(detId, true));
	subDetector_trk.insert(std::pair<unsigned int,
			       int>( detId,((*t_DetIds)[idet] &0xe000000) / 0x2000000 ));
      }
      
    }

// --- Fill initial histograms ---------------------------      
    e2p_init->Fill(response ,1.0);

    if ( abs_t_ieta < FIRST_IETA_TR )
      e2pHB_init->Fill(response ,1.0);
    else if ( abs_t_ieta < FIRST_IETA_HE )
      e2pTR_init->Fill(response ,1.0);
    else
      e2pHE_init->Fill(response ,1.0);

    if ( debug > 1 ) {
      std::cout << "***Entry : " << ientry
		<< " ***ieta/p/Ecal/nDet : "
		<< t_ieta << "/" << t_p
		<< "/" << t_eMipDR << "/" << (*t_DetIds).size() 
		<< " ***Etot/E10/E30/Ecor/cPU : " << t_eHcal
		<< "/" << t_eHcal10 << "/" << t_eHcal30
		<< "/" << eTotalCor << "/" << correctionForPU
		<< "(" << de2p << ")"
		<< std::endl;
    }
    if ( response > maxRespForGoodTrack  )
      maxRespForGoodTrack = response;
    if ( response < minRespForGoodTrack )
      minRespForGoodTrack = response;
    if ( response > MAX_RESPONSE_HIST )
      nRespOverHistLimit++;

    if ( response < LOW_RESPONSE ) ieta_lefttail->Fill(t_ieta);
    if ( response > HIGH_RESPONSE ) ieta_righttail->Fill(t_ieta);

    int jj = HALF_NUM_ETA_BINS + int(t_ieta/N_ETA_RINGS_PER_BIN) + (t_ieta>0);
    ntrk_ieta[jj]++;
    
  } // ------------------- end of loop over events -------------------------------------

  for ( int j = 0; j < NUM_ETA_BINS; j++ ) {
    if ( maxNumOfTracksForIeta < ntrk_ieta[j] ) maxNumOfTracksForIeta = ntrk_ieta[j];
  }

  double jeta[N_DEPTHS][MAXNUM_SUBDET];
  double nTrk[N_DEPTHS][MAXNUM_SUBDET];
  double nSub[N_DEPTHS][MAXNUM_SUBDET];
  double nPhi[N_DEPTHS][MAXNUM_SUBDET];
  double rms[N_DEPTHS][MAXNUM_SUBDET];
  unsigned int kdep[N_DEPTHS];
  for ( unsigned ik = 0; ik < N_DEPTHS; ik++ ) { kdep[ik] = 0; }

  // fill number of tracks
  std::map <unsigned int,int>::iterator nTrksItr = nTrks.begin();
  for (nTrksItr = nTrks.begin(); nTrksItr != nTrks.end(); nTrksItr++ ) {
    unsigned int detId = nTrksItr->first;
    int depth= ((detId>>DEPTH_OFFSET) & DEPTH_MASK) + int(MERGE_PHI_AND_DEPTHS);
    int zside= (detId&ZSIDE_MASK) ? 1 : -1;
    unsigned int kcur = kdep[depth-1];
    
    jeta[depth-1][kcur] = int((detId>>ETA_OFFSET) & ETA_MASK)*zside;
    nTrk[depth-1][kcur] = nTrksItr->second;
    nSub[depth-1][kcur] = double(nSubdetInEvent[detId])/double(nTrksItr->second);
    nPhi[depth-1][kcur] = double(nPhiMergedInEvent[detId])/double(nTrksItr->second);
    if ( nTrk[depth-1][kcur] > 1 ) 
      rms[depth-1][kcur] = sqrt((sumOfResponseSquared[detId] -
				 pow(sumOfResponse[detId],2)/nTrk[depth-1][kcur])
				/(nTrk[depth-1][kcur] - 1));
    else rms[depth-1][kcur] = RESOLUTION_HCAL;
    kdep[depth-1]++;
  }
  for ( unsigned ik = 0; ik < N_DEPTHS; ik++ ) {
    double x[MAXNUM_SUBDET];
    double ytrk[MAXNUM_SUBDET], ysub[MAXNUM_SUBDET];
    double yphi[MAXNUM_SUBDET], yrms[MAXNUM_SUBDET];
    for ( unsigned im = 0; im < MAXNUM_SUBDET; im++ ) {
      x[im] = jeta[ik][im];
      ytrk[im] = nTrk[ik][im];
      ysub[im] = nSub[ik][im];
      yphi[im] = nPhi[ik][im];
      yrms[im] = rms[ik][im];
    }
    TGraph*  g_ntrk = new TGraph(kdep[ik], x, ytrk);
    sprintf(name, "Number of tracks for depth %1d", ik+1);
    g_ntrk->SetTitle(name);
    sprintf(name, "nTrk_depth%1d", ik+1);
    foutRootFile->WriteTObject(g_ntrk, name);

    TGraph*  g_nsub = new TGraph(kdep[ik], x, ysub);
    sprintf(name, "Mean number of active subdetectors, depth %1d", ik+1);
    g_nsub->SetTitle(name);
    sprintf(name, "nSub_depth%1d", ik+1);
    foutRootFile->WriteTObject(g_nsub, name);

    TGraph*  g_nphi = new TGraph(kdep[ik], x, yphi);
    sprintf(name, "Mean number of phi-merged subdetectors, depth %1d", ik+1);
    g_nphi->SetTitle(name);
    sprintf(name, "nPhi_depth%1d", ik+1);
    foutRootFile->WriteTObject(g_nphi, name);

    TGraph*  g_rms = new TGraph(kdep[ik], x, yrms);
    sprintf(name, "RMS of samples, depth %1d", ik+1);
    g_rms->SetTitle(name);
    sprintf(name, "rms_depth%1d", ik+1);
    foutRootFile->WriteTObject(g_rms, name);
  }

  //--- estimate ratio mean/MPV
  double xl = e2p_init->GetMean() - FIT_RMS_INTERVAL*e2p_init->GetRMS();
  double xr = e2p_init->GetMean() + FIT_RMS_INTERVAL*e2p_init->GetRMS();
  e2p_init->Fit("f1","QN", "R", xl, xr);
  xl = f1->GetParameter(1) - FIT_RMS_INTERVAL*f1->GetParameter(2);
  xr = f1->GetParameter(1) + FIT_RMS_INTERVAL*f1->GetParameter(2);
  e2p_init->Fit("f1","QN", "R", xl, xr);

  if ( shiftResp && (f1->GetParameter(1) != 0) ) {
    referenceResponse = e2p_init->GetMean()/f1->GetParameter(1);
    std::cout << "Use reference response=<mean from sample>/<mpv from fit>:"
	      << e2p_init->GetMean() << "/" << f1->GetParameter(1)
	      << " = " << referenceResponse //<< std::endl
	      << " (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }
  else {
    referenceResponse = 1;
    std::cout << "Use reference response = 1" << std::endl
	      << "<mean from sample>/<mpv from fit> = "
	      << e2p_init->GetMean()/f1->GetParameter(1)
	      << "  (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }
  //---- for HB
  xl = e2pHB_init->GetMean() - FIT_RMS_INTERVAL*e2pHB_init->GetRMS();
  xr = e2pHB_init->GetMean() + FIT_RMS_INTERVAL*e2pHB_init->GetRMS();
  e2pHB_init->Fit("f1","QN", "R", xl, xr);
  xl = f1->GetParameter(1) - FIT_RMS_INTERVAL*f1->GetParameter(2);
  xr = f1->GetParameter(1) + FIT_RMS_INTERVAL*f1->GetParameter(2);
  e2pHB_init->Fit("f1","QN", "R", xl, xr);

  if ( shiftResp && (f1->GetParameter(1) != 0) ) {
    referenceResponseHB = e2pHB_init->GetMean()/f1->GetParameter(1);
    std::cout << "In HB <mean from sample>/<mpv from fit> = "
	      << e2pHB_init->GetMean() << "/" << f1->GetParameter(1)
	      << " = " << referenceResponseHB //<< std::endl
	      << " (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }
  else {
    referenceResponseHB = 1;
    std::cout << "Use reference response in HB = 1" << std::endl
	      << "<mean from sample>/<mpv from fit> = "
	      << e2pHB_init->GetMean()/f1->GetParameter(1)
	      << "  (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }
  //---- for TR
  xl = e2pTR_init->GetMean() - FIT_RMS_INTERVAL*e2pTR_init->GetRMS();
  xr = e2pTR_init->GetMean() + FIT_RMS_INTERVAL*e2pTR_init->GetRMS();
  e2pTR_init->Fit("f1","QN", "R", xl, xr);
  xl = f1->GetParameter(1) - FIT_RMS_INTERVAL*f1->GetParameter(2);
  xr = f1->GetParameter(1) + FIT_RMS_INTERVAL*f1->GetParameter(2);
  e2pTR_init->Fit("f1","QN", "R", xl, xr);

  if ( shiftResp && (f1->GetParameter(1) != 0) ) {
    referenceResponseTR = e2pTR_init->GetMean()/f1->GetParameter(1);
    std::cout << "In TR <mean from sample>/<mpv from fit> = "
	      << e2pTR_init->GetMean() << "/" << f1->GetParameter(1)
	      << " = " << referenceResponseTR //<< std::endl
	      << " (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }
  else {
    referenceResponseTR = 1;
    std::cout << "Use reference response in TR = 1" << std::endl
	      << "<mean from sample>/<mpv from fit> = "
	      << e2pTR_init->GetMean()/f1->GetParameter(1)
	      << "  (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }
  //---- for HE
  xl = e2pHE_init->GetMean() - FIT_RMS_INTERVAL*e2pHE_init->GetRMS();
  xr = e2pHE_init->GetMean() + FIT_RMS_INTERVAL*e2pHE_init->GetRMS();
  e2pHE_init->Fit("f1","QN", "R", xl, xr);
  xl = f1->GetParameter(1) - FIT_RMS_INTERVAL*f1->GetParameter(2);
  xr = f1->GetParameter(1) + FIT_RMS_INTERVAL*f1->GetParameter(2);
  e2pHE_init->Fit("f1","QN", "R", xl, xr);

  if ( shiftResp && (f1->GetParameter(1) != 0) ) {
    referenceResponseHE = e2pHE_init->GetMean()/f1->GetParameter(1);
    std::cout << "In HE <mean from sample>/<mpv from fit> = "
	      << e2pHE_init->GetMean() << "/" << f1->GetParameter(1)
	      << " = " << referenceResponseHE //<< std::endl
	      << " (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }
  else {
    referenceResponseHE = 1;
    std::cout << "Use reference response in HE = 1" << std::endl
	      << "<mean from sample>/<mpv from fit> = "
	      << e2pHE_init->GetMean()/f1->GetParameter(1)
	      << "  (chi2ndf = " << f1->GetChisquare()/f1->GetNDF() << ")"
	      << std::endl;
  }

  //----- print additional info
  std::cout << "Maximal response for good tracks = " 
	    << maxRespForGoodTrack << std::endl
	    << nRespOverHistLimit
	    << " events with response > " << MAX_RESPONSE_HIST
	    << "(hist limit for mean estimate)"
	    << std::endl;
  std::cout << "Minimal response for good tracks = " 
	    << minRespForGoodTrack
	    << std::endl;
  std::cout << "Maximum number of selected tracks per ieta bin = " 
	    << maxNumOfTracksForIeta
	    << std::endl;
  std::cout << "Number of selected tracks in HB = "
	    << e2pHB_init->GetEntries()
	    << std::endl;
  std::cout << "Number of selected tracks in TR = "
	    << e2pTR_init->GetEntries()
	    << std::endl;
  std::cout << "Number of selected tracks in HE = "
	    << e2pHE_init->GetEntries()
	    << std::endl;

  /*
  xl = e2pHB_init->GetMean() - FIT_RMS_INTERVAL*e2pHB_init->GetRMS();
  xr = e2pHB_init->GetMean() + FIT_RMS_INTERVAL*e2pHB_init->GetRMS();
  e2pHB_init->Fit("f1","QN", "R", xl, xr);

  xl = e2pHE_init->GetMean() - FIT_RMS_INTERVAL*e2pHE_init->GetRMS();
  xr = e2pHE_init->GetMean() + FIT_RMS_INTERVAL*e2pHE_init->GetRMS();
  e2pHE_init->Fit("f1","QN", "R", xl, xr);
  */
  return nSelectedEvents;
}

//**********************************************************
// Loop over events in the tree for current iteration
//**********************************************************
Double_t CalibTree::loopForIteration(unsigned int subsample,
				     unsigned int nIter,
				     unsigned int debug )
{
  char name[500];
  double meanDeviation = 0;
  unsigned int ndebug(0);
    
  TF1* f1 = new TF1("f1","gaus", MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  TH1F* e2p[NUM_ETA_BINS];

  int n_ieta_bins = 2*2.5*pow(maxNumOfTracksForIeta,1/3.0);
  for ( int i = 0; i < NUM_ETA_BINS; i++ ) {
    sprintf(name,"e2p[%02d]", i);
    e2p[i] = new TH1F(name, "",
		      n_ieta_bins, //NBIN_RESPONSE_HIST_IND,
		      MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
    e2p[i]->Sumw2();
  }

  std::map<unsigned int, std::pair<double,double> > sumsForFactorCorrection;
  std::map<unsigned int, double> sumOfWeightsSquared;

  if ( debug > 0 ) {
    std::cout.precision(3);
    std::cout << "-------------------------------------------- nIter = "
	      << nIter << std::endl;
  }
//--- initialize chain ----------------------------------------
  if (fChain == 0) return 0;
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nb = 0;
  
// ----------------------- loop over events -------------------------------------  
  for (Long64_t jentry=0; jentry<nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if ( ientry < 0 || ndebug > debug ) break;   
    nb = fChain->GetEntry(jentry);   //nbytes += nb;
    
    if ( (jentry%2 == subsample) ) continue;   // only odd or even events

    // --------------- selection of good track --------------------
    
    if ( !goodTrack(t_ieta) ) continue;

    if ( debug > 1 ) {
      ndebug++;
      std::cout << "***Entry (Track) Number : " << ientry 
		<< " p/eHCal/eMipDR/nDets : " << t_p << "/" << t_eHcal 
		<< "/" << t_eMipDR << "/" << (*t_DetIds).size() 
		<< std::endl;
    }
    
    double eTotal(0.0);
    double eTotalWithEcal(0.0);
      
    // ---- first loop over active subdetectors in the event for total energy ---

    for (unsigned int idet = 0; idet < (*t_DetIds).size(); idet++) { 
      double hitEnergy(0);	
      unsigned int detId = ( (*t_DetIds)[idet] & MASK ) | MASK2 ;
	
      if (factors.find(detId) != factors.end()) 
	hitEnergy = factors[detId] * (*t_HitEnergies)[idet];
      else 
	hitEnergy = (*t_HitEnergies)[idet];

      eTotal += hitEnergy;
    }

    eTotalWithEcal = eTotal + t_eMipDR;    

// --- Correction for PU   --------      
    double eTotalCor(eTotal);
    double eTotalWithEcalCor(eTotalWithEcal);
    //double e10(0.0);
    //double e30(0.0);
    double correctionForPU(1.0);

    if ( APPLY_CORRECTION_FOR_PU ) {
      /*
      for (unsigned int idet1 = 0; idet1 < (*t_DetIds1).size(); idet1++) { 
	double hitEnergy(0);
	unsigned int detId1 = ( (*t_DetIds1)[idet1] & MASK ) | MASK2;
	
	if (factors.find(detId1) != factors.end()) 
	  hitEnergy = factors[detId1] * (*t_HitEnergies1)[idet1];
	else 
	  hitEnergy = (*t_HitEnergies1)[idet1];
	
	e10 += hitEnergy;
      }
      for (unsigned int idet3 = 0; idet3 < (*t_DetIds3).size(); idet3++) { 
	double hitEnergy(0);
	unsigned int detId3 = ( (*t_DetIds3)[idet3] & MASK ) | MASK2;
	
	if (factors.find(detId3) != factors.end()) 
	  hitEnergy = factors[detId3] * (*t_HitEnergies3)[idet3];
	else
	  hitEnergy = (*t_HitEnergies3)[idet3];
	
	e30 += hitEnergy;
      }
      double de2p = (e30 - e10)/t_p;
      */
      double de2p = (t_eHcal30 - t_eHcal10)/t_p;
      if ( de2p > DELTA_CUT ) {
	int abs_t_ieta = abs(t_ieta);
	int icor = int(abs_t_ieta >= FIRST_IETA_TR) + int(abs_t_ieta >= FIRST_IETA_HE)
	  + int(abs_t_ieta >= FIRST_IETA_FWD_1) + int(abs_t_ieta >= FIRST_IETA_FWD_2);
	correctionForPU = (1 + LINEAR_COR_COEF[icor]*(t_eHcal/t_p)*de2p
			   *(1 + SQUARE_COR_COEF[icor]*de2p));
      }
    }    

    // check for possibility to correct for PU
    if ( correctionForPU <= 0 || correctionForPU > 1 ) continue;

    eTotalCor = eTotal*correctionForPU;
    eTotalWithEcalCor = eTotalCor + t_eMipDR;
     
    int jeta = HALF_NUM_ETA_BINS + int(t_ieta/N_ETA_RINGS_PER_BIN) + (t_ieta>0);
    e2p[jeta]->Fill(eTotalWithEcalCor/t_p ,1.0);
      
// ---- second loop over active subdetectors in the event  -----------------
      
    double response = eTotalWithEcalCor/t_p; // - referenceResponse;
    
    for (unsigned int idet = 0; idet < (*t_DetIds).size(); idet++) {
      double hitEnergy(0);
      unsigned int detId = ( (*t_DetIds)[idet] & MASK ) | MASK2 ;
		 
      if (factors.find(detId) != factors.end())
	hitEnergy = factors[detId] * (*t_HitEnergies)[idet];
      else
	hitEnergy = (*t_HitEnergies)[idet];

      double cellWeight = hitEnergy/eTotal;   
      //double trackWeight = (cellWeight * t_p) / eTotalWithEcalCor; // old method
      double trackWeight = cellWeight*response;   // new method
      double cellweight2 = cellWeight*cellWeight;
      
      if( sumsForFactorCorrection.find(detId) != sumsForFactorCorrection.end() ) {
	cellWeight  += sumsForFactorCorrection[detId].first;
	trackWeight += sumsForFactorCorrection[detId].second;
	sumsForFactorCorrection[detId] = std::pair<double,double>(cellWeight,trackWeight);
	sumOfWeightsSquared[detId] += cellweight2;
      }
      else {
	sumsForFactorCorrection.insert(std::pair<unsigned int,
				       std::pair<double,double> >(detId,
								  std::pair<double,double>(cellWeight,
											   trackWeight)));
	sumOfWeightsSquared.insert(std::pair<unsigned int,double>(detId, cellweight2));
     }
	
      if ( debug > 1 ) { //|| hitEnergy < -0.5) {
	double f = 1;
	int zside= (detId&ZSIDE_MASK) ? 1 : -1;
	if (factors.find(detId) != factors.end()) f = factors[detId];
	std::cout << jentry << "::: "
		  << " Ncells: " << (*t_DetIds).size()
		  << " !! detId(ieta)/e/f : " 
	    //    << std::hex << (*t_DetIds)[idet] << ":"
		  << detId << "(" << int((detId>>ETA_OFFSET) & ETA_MASK)*zside << ")"
		  << "/" << hitEnergy
		  << "/" << f
		  << " ||| cellW/trW : " << cellWeight << " / " << trackWeight
		  << " ||| E/Ecor/p : " << eTotal
		  << " / " << eTotalCor
		  << " / " << t_p
		  << " || e10/e30/cF : " << t_eHcal10
		  << " / " << t_eHcal30
		  << " / " << correctionForPU
		  << std::endl;
      }
    }  // --------------- end of second loop over cells ----------
  } // ------------------- end of loop over events -------------------------------------

//----- Graphs to be saved in root file ----------------
  if ( debug > 0 ) {
    std::cout << "Fit and calculate means..." << std::endl;
    std::cout << "Number of plots (ieta bins) = " << NUM_ETA_BINS << std::endl;
  }
  TGraph *g_chi = new TGraph(NUM_ETA_BINS);  
  TGraphErrors* g_e2pFit = new TGraphErrors(NUM_ETA_BINS);
  TGraphErrors* g_e2pMean = new TGraphErrors(NUM_ETA_BINS);
  TGraph *g_nhistentries = new TGraph(NUM_ETA_BINS);
  
  int ipoint(0);
  for ( int i = 0; i < NUM_ETA_BINS; i++ ) {
    int ieta = (i - HALF_NUM_ETA_BINS - (i>HALF_NUM_ETA_BINS))*N_ETA_RINGS_PER_BIN;
    if ( N_ETA_RINGS_PER_BIN > 1 ) {
      ieta = (i > HALF_NUM_ETA_BINS) ? ieta+1 : ieta-1;
    }
    int nhistentries = e2p[i]->GetEntries();
    /*
      if ( debug > 0 ) {
	std::cout << "i / entries / ieta :::"
		  << i
		  << " / " << nhistentries
		  << " / " << ieta
		  << std::endl;
      }
    */
     if ( nIter == 1 ) {
       g_nhistentries->SetPoint(i, ieta, nhistentries);
     }

    if ( nhistentries < 1 ) continue;
    else {
      g_e2pMean->SetPoint(ipoint, ieta, e2p[i]->GetMean());
      g_e2pMean->SetPointError(ipoint, 0, e2p[i]->GetMeanError());

      if ( nhistentries > MIN_N_ENTRIES_FOR_FIT ) {
	int nrebin = n_ieta_bins/(2*2.5*pow(nhistentries,1/3.0));
	if ( nrebin > 2 ) e2p[i]->Rebin(nrebin);
	
	double xl = e2p[i]->GetMean() - FIT_RMS_INTERVAL*e2p[i]->GetRMS();
	double xr = e2p[i]->GetMean() + FIT_RMS_INTERVAL*e2p[i]->GetRMS();
	e2p[i]->Fit("f1","QN", "R", xl, xr);
	xl = f1->GetParameter(1) - FIT_RMS_INTERVAL*f1->GetParameter(2);
	xr = f1->GetParameter(1) + FIT_RMS_INTERVAL*f1->GetParameter(2);
	e2p[i]->Fit("f1","QN", "R", xl, xr);
	g_e2pFit->SetPoint(ipoint, ieta, f1->GetParameter(1));
	g_e2pFit->SetPointError(ipoint, 0, f1->GetParError(1));
	g_chi->SetPoint(ipoint, ieta, f1->GetChisquare()/f1->GetNDF());
      }
      else {
	g_e2pFit->SetPoint(ipoint, ieta, e2p[i]->GetMean());
	g_e2pFit->SetPointError(ipoint, 0, e2p[i]->GetMeanError());
	g_chi->SetPoint(ipoint, ieta, 0);
      }
      ipoint++;
    }
  }
  // fill number of tracks per ieta
  if ( nIter == 1 ) {
      sprintf(name, "Number of selected tracks");
      g_nhistentries->SetTitle(name);
      g_nhistentries->GetXaxis()->SetTitle("i#eta");
      sprintf(name, "selTrks");
      foutRootFile->WriteTObject(g_nhistentries, name);   
  }

  for ( int k = ipoint; k < NUM_ETA_BINS; k++ ) {
    g_e2pFit->RemovePoint(ipoint);
    g_e2pMean->RemovePoint(ipoint);
  }
  sprintf(name, "Response from fit, iteration %2d", nIter);
  g_e2pFit->SetTitle(name);
  g_e2pFit->GetXaxis()->SetTitle("i#eta");
  sprintf(name, "respFit_%d", nIter);
  foutRootFile->WriteTObject(g_e2pFit, name);

  sprintf(name, "Mean response, iteration %2d", nIter);
  g_e2pMean->SetTitle(name);
  g_e2pMean->GetXaxis()->SetTitle("i#eta");
  sprintf(name, "respMean_%d", nIter);
  foutRootFile->WriteTObject(g_e2pMean, name);

  sprintf(name, "Chi2/NDF, iteration %2d", nIter);
  g_chi->SetTitle(name);
  g_chi->GetXaxis()->SetTitle("i#eta");
  sprintf(name, "chi2ndf_%d", nIter);
  foutRootFile->WriteTObject(g_chi, name);

// --- convergence criteria and correction factors -----------------------------------

  double MeanConvergenceDelta(0),  MaxRelDeviationWeights(0), MaxRatioUncertainties(0);
  double dets[MAXNUM_SUBDET];
  double ztest[MAXNUM_SUBDET], sys2statRatio[MAXNUM_SUBDET];

  if ( debug > 0 ) std::cout << "Calculate correction factors..." << std::endl;

  unsigned int kount(0), mkount(0);
  unsigned int maxKountW(0), maxKountR(0);

  double fac[N_DEPTHS][MAXNUM_SUBDET];
  double dfac[N_DEPTHS][MAXNUM_SUBDET];
  double ieta[N_DEPTHS][MAXNUM_SUBDET];
  double dieta[N_DEPTHS][MAXNUM_SUBDET];

  unsigned int kdep[N_DEPTHS];
  for ( unsigned ik = 0; ik < N_DEPTHS; ik++ ) { kdep[ik] = 0; }

//-------------- loop over all cells ---------------------------------
  std::map <unsigned int,
	    std::pair<double,double> >::iterator sumsForFactorCorrectionItr
    = sumsForFactorCorrection.begin();
  for (; sumsForFactorCorrectionItr != sumsForFactorCorrection.end();
       sumsForFactorCorrectionItr++) {

    unsigned int detId = sumsForFactorCorrectionItr->first;
    double sumOfWeights = (sumsForFactorCorrectionItr->second).first;
    int nSubDetTracks(0);
    double subdetRMS(RESOLUTION_HCAL);
    if ( nTrks.find(detId) != nTrks.end() ) {
      nSubDetTracks = nTrks[detId];
      if ( nSubDetTracks > 1 ) 
	subdetRMS = sqrt((sumOfResponseSquared[detId] 
			  - pow(sumOfResponse[detId],2)/double(nSubDetTracks))
			 /double(nSubDetTracks - 1));
    }
    else {
      std::cout << "!!!!!!! No tracks for subdetector " << detId << std::endl;
      continue;
    }
    double NcellMean = double(nSubdetInEvent[detId])/double(nSubDetTracks);

    double ratioWeights(1);
    if ( abs(sumOfWeights) > 0 )
      ratioWeights = sqrt(sumOfWeightsSquared[detId])/sumOfWeights;
    double correctionRMS = subdetRMS*ratioWeights*sqrt(NcellMean);
    
    double absErrorW(0);
    double absErrorWprevious(0);
    double factorPrevious(1);
    double factorCorrection(1);

    int zside = (detId&ZSIDE_MASK) ? 1 : -1;
    int depth = ((detId>>DEPTH_OFFSET)&DEPTH_MASK) + int(MERGE_PHI_AND_DEPTHS);
    unsigned int kcur = kdep[depth-1];
    
    ieta[depth-1][kcur] = int((detId>>ETA_OFFSET) & ETA_MASK)*zside;
    dieta[depth-1][kcur] = 0;
    
    double refR = referenceResponse;
    if ( !SINGLE_REFERENCE_RESPONSE ) {
      if ( abs(ieta[depth-1][kcur]) < FIRST_IETA_TR )
	refR = referenceResponseHB;
      else if ( abs(ieta[depth-1][kcur]) < FIRST_IETA_HE )
	refR = referenceResponseTR;
      else
	refR = referenceResponseHE;
    }
	
    //--- old expression ---------------
    /*
    factorCorrection = (sumsForFactorCorrectionItr->second).second
                     / (sumsForFactorCorrectionItr->second).first;
    */
    //--- new expression --------------
    if ( abs(sumOfWeights) > 0 )  
      factorCorrection = 1 + refR
	- (sumsForFactorCorrectionItr->second).second / sumOfWeights;
    //---------------------------------
    if ( correctionRMS/factorCorrection > MAX_REL_UNC_FACTOR ||
	 nSubDetTracks < MIN_N_TRACKS_PER_CELL ) {
      correctionRMS = sqrt(pow(correctionRMS,2) + pow((factorCorrection - 1),2));
      factorCorrection = 1;
    }
    
    if( nSubDetTracks > MIN_N_TRACKS_PER_CELL ) {
      if (factorCorrection > 1) MeanConvergenceDelta += (1 - 1/factorCorrection);
      else                      MeanConvergenceDelta += (1 - factorCorrection);
      mkount++;
    }

              
    if (factors.find(detId) != factors.end()) {
      factorPrevious = factors[detId];
      factors[detId] *= factorCorrection;
      absErrorWprevious = uncFromWeights[detId];
      absErrorW = factorPrevious*correctionRMS;
      uncFromWeights[detId] = absErrorW;
      uncFromDeviation[detId] = factorPrevious*abs(factorCorrection - 1);
    }
    else {
      factorPrevious = 1;
      factors.insert(std::pair<unsigned int, double>(detId, factorCorrection));
      subDetector_final.insert(std::pair<unsigned int, double>(detId,
							       subDetector_trk[detId]));
      absErrorW = correctionRMS;
      absErrorWprevious = 0;
      uncFromWeights.insert(std::pair<unsigned int, double>(detId, absErrorW));
      uncFromDeviation.insert(std::pair<unsigned int, double>(detId,
							      abs(factorCorrection - 1)));
    }

    if ( debug > 0 ) {
      //if ( ieta[depth-1][kcur] == 27 && depth == 2 ) {
      std::cout.precision(3);
      std::cout << detId // << " (" << mkount << ")"
	        << " *** ieta/depth | rw | cw | tw | fCor | nTrk | Ncell | C |::: "
	        << ieta[depth-1][kcur] << "/" << depth << " | "
		<< ratioWeights << " | "
	        << sumOfWeights << " | "
	        << (sumsForFactorCorrectionItr->second).second << " | "
	        << factorCorrection << " | "
		<< nSubDetTracks << " | "
		<< NcellMean << " | "
		<< correctionRMS << " |"
	        << std::endl;
    }

    dets[kount] = detId;

    fac[depth-1][kcur] = factors[detId];
    dfac[depth-1][kcur] =
      sqrt(pow(uncFromWeights[detId],2) + pow(uncFromDeviation[detId],2));
    
    sys2statRatio[kount] = abs(factorPrevious*(factorCorrection - 1))/absErrorW;
    if ( sys2statRatio[kount] > MaxRatioUncertainties ) {
      MaxRatioUncertainties = sys2statRatio[kount];
      maxKountR = kount;
    }
    ztest[kount] = factorPrevious*(factorCorrection - 1)
      /sqrt(pow(absErrorWprevious,2) + pow(absErrorW,2));
    if ( abs(ztest[kount]) > MaxRelDeviationWeights ) {
      MaxRelDeviationWeights = abs(ztest[kount]);
      maxKountW = kount;
    } 
    kount++;
    kdep[depth-1]++;
  }

//---- write current plots -----------------------------
  if ( debug > 0 ) std::cout << "Write graphs..." << std::endl;

  for ( unsigned ik = 0; ik < N_DEPTHS; ik++ ) {
    double x[MAXNUM_SUBDET], dx[MAXNUM_SUBDET], y[MAXNUM_SUBDET], dy[MAXNUM_SUBDET];
    for ( unsigned im = 0; im < MAXNUM_SUBDET; im++ ) {
      x[im] = ieta[ik][im];
      dx[im] = dieta[ik][im];
      y[im] = fac[ik][im];
      dy[im] = dfac[ik][im];
    }
    TGraphErrors*  g_fac = new TGraphErrors(kdep[ik], x, y, dx, dy);
    sprintf(name, "Extracted correction factors, depth %1d", ik+1);
    g_fac->SetTitle(name);
    g_fac->GetXaxis()->SetTitle("i#eta");
    sprintf(name, "Cfacs_depth%1d_%02d", ik+1, nIter);
    foutRootFile->WriteTObject(g_fac, name);
  }

  TGraph  *g_ztest, *g_sys2stat;

  g_ztest = new TGraph(kount, dets, ztest); 
  sprintf(name, "Z-test (unc. from weights) vs detId for iter %d", nIter);
  g_ztest->SetTitle(name);
  sprintf(name, "Ztest_detId_%02d", nIter);
  foutRootFile->WriteTObject(g_ztest, name);

  g_sys2stat = new TGraph(kount, dets, sys2statRatio); 
  sprintf(name, "Ratio of syst. to stat. unc. vs detId for iter %d", nIter);
  g_sys2stat->SetTitle(name);
  sprintf(name, "Sys2stat_detId_%02d", nIter);
  foutRootFile->WriteTObject(g_sys2stat, name);

  std::cout << "----------Iteration " << nIter << "--------------------" << std::endl;
  maxZtestFromWeights = MaxRelDeviationWeights; 
  std::cout << "Max abs(Z-test) with stat errors from weights = "
	    << maxZtestFromWeights << " for subdetector " << maxKountW << std::endl;
  maxSys2StatRatio = MaxRatioUncertainties; 
  std::cout << "Max ratio of syst.(f_cur - f_prev) to stat. uncertainty = "
	    << maxSys2StatRatio << " for subdetector " << maxKountR << std::endl;

  meanDeviation = (mkount > 0) ? (MeanConvergenceDelta/mkount) : 0;
  std::cout << "Mean absolute deviation from previous iteration = " << meanDeviation
	    << " for " << mkount
	    << " from " << kount << " DetIds" << std::endl;

//--- delete hists ---------------------------
  for ( int i = 0; i < NUM_ETA_BINS; i++ ) {
    delete e2p[i];
  }

  return meanDeviation;
}
//**********************************************************
// Last loop over events in the tree
//**********************************************************
Double_t CalibTree::lastLoop(unsigned int subsample,
			     unsigned int maxIter,
			     bool isTest,
			     unsigned int debug)
{
  char name[100];
  unsigned int ndebug(0);
  
  char stest[80] = "test";
  if ( !isTest )
    sprintf(stest,"after %2d iterations", maxIter);
  char scorr[80] = "correction for PU";
  char sxlabel[80] ="(E^{cor}_{hcal} + E_{ecal})/p_{track}"; 
  if ( !APPLY_CORRECTION_FOR_PU ) {
    sprintf(scorr,"no correction for PU");
    sprintf(sxlabel,"(E_{hcal} + E_{ecal})/p_{track}");
  } 
    
  TF1* f1 = new TF1("f1","gaus", MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);

  sprintf(name,"HB+HE: %s, %s", stest, scorr);
  e2p_last = new TH1F("e2p_last", name,
		      NBIN_RESPONSE_HIST, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2p_last->Sumw2();
  e2p_last->GetXaxis()->SetTitle(sxlabel);
  
  sprintf(name,"HB: %s, %s", stest, scorr);
  e2pHB_last = new TH1F("e2pHB_last", name,
			NBIN_RESPONSE_HIST/2, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2pHB_last->Sumw2();
  e2pHB_last->GetXaxis()->SetTitle(sxlabel);

  sprintf(name,"Initial TR: %s", scorr);
  e2pTR_last = new TH1F("e2pTR_last", name,
			NBIN_RESPONSE_HIST/10, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2pTR_last->Sumw2();
  e2pTR_last->GetXaxis()->SetTitle(sxlabel);
  
  sprintf(name,"HE: %s, %s", stest, scorr);
  e2pHE_last = new TH1F("e2pHE_last", name,
			NBIN_RESPONSE_HIST/2, MIN_RESPONSE_HIST, MAX_RESPONSE_HIST);
  e2pHE_last->Sumw2();
  e2pHE_last->GetXaxis()->SetTitle(sxlabel);
 
//--- initialize chain ----------------------------------------
  if (fChain == 0) return 0;
  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nb = 0;
  
  int nSelectedEvents(0);

  if ( debug > 0 ) { 
    std::cout << "------------- Last loop after " << maxIter << " iterations"
	      << std::endl;
  }
// ----------------------- loop over events -------------------------------------  
  for (Long64_t jentry=0; jentry<nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if ( ientry < 0 || ndebug > debug ) break;   
    nb = fChain->GetEntry(jentry);   //nbytes += nb;
    
    if ( (jentry%2 == subsample) ) continue;   // only odd or even events

    // --------------- selection of good track --------------------
    
    if ( !goodTrack(t_ieta) ) continue;

    nSelectedEvents++;
    
    if ( debug > 1 ) {
      ndebug++;
      std::cout << "***Entry (Track) Number : " << ientry 
		<< " p/eHCal/eMipDR/nDets : " << t_p << "/" << t_eHcal 
		<< "/" << t_eMipDR << "/" << (*t_DetIds).size() 
		<< std::endl;
    }
    
    double eTotal(0.0);
    double eTotalWithEcal(0.0);
      
    // ---- loop over active cells in the event for total energy ---

    for (unsigned int idet = 0; idet < (*t_DetIds).size(); idet++) { 
      double hitEnergy(0);	
      unsigned int detId = ( (*t_DetIds)[idet] & MASK ) | MASK2 ;
	
      if (factors.find(detId) != factors.end()) 
	hitEnergy = factors[detId] * (*t_HitEnergies)[idet];
      else 
	hitEnergy = (*t_HitEnergies)[idet];

      eTotal += hitEnergy;
    }

    eTotalWithEcal = eTotal + t_eMipDR;    

// --- Correction for PU   --------      
    double eTotalCor(eTotal);
    double eTotalWithEcalCor(eTotalWithEcal);
    //double e10(0.0);
    //double e30(0.0);
    double correctionForPU(1.0);
    int abs_t_ieta = abs(t_ieta);

    if ( APPLY_CORRECTION_FOR_PU ) { 
      /* 
      for (unsigned int idet1 = 0; idet1 < (*t_DetIds1).size(); idet1++) { 
	double hitEnergy(0);
	unsigned int detId1 = ( (*t_DetIds1)[idet1] & MASK ) | MASK2;
	
	if (factors.find(detId1) != factors.end()) 
	  hitEnergy = factors[detId1] * (*t_HitEnergies1)[idet1];
	else 
	  hitEnergy = (*t_HitEnergies1)[idet1];
	
	e10 += hitEnergy;
      }
      for (unsigned int idet3 = 0; idet3 < (*t_DetIds3).size(); idet3++) { 
	double hitEnergy(0);
	unsigned int detId3 = ( (*t_DetIds3)[idet3] & MASK ) | MASK2;
	
	if (factors.find(detId3) != factors.end()) 
	  hitEnergy = factors[detId3] * (*t_HitEnergies3)[idet3];
	else
	  hitEnergy = (*t_HitEnergies3)[idet3];
	
	e30 += hitEnergy;
      }
      double de2p = (e30 - e10)/t_p;
      */
      
      double de2p = (t_eHcal30 - t_eHcal10)/t_p;
      if ( de2p > DELTA_CUT ) {
	int icor = int(abs_t_ieta >= FIRST_IETA_TR) + int(abs_t_ieta >= FIRST_IETA_HE)
	  + int(abs_t_ieta >= FIRST_IETA_FWD_1) + int(abs_t_ieta >= FIRST_IETA_FWD_2);
	correctionForPU = (1 + LINEAR_COR_COEF[icor]*(t_eHcal/t_p)*de2p
			   *(1 + SQUARE_COR_COEF[icor]*de2p));
      }
    }      
    // check for possibility to correct for PU
    if ( correctionForPU <= 0 || correctionForPU > 1 ) continue;

    eTotalCor = eTotal*correctionForPU;
    eTotalWithEcalCor = eTotalCor + t_eMipDR;
      
    e2p_last->Fill(eTotalWithEcalCor/t_p ,1.0);

    if ( abs_t_ieta < FIRST_IETA_TR )
      e2pHB_last->Fill(eTotalWithEcalCor/t_p ,1.0);
    else if ( abs_t_ieta < FIRST_IETA_HE )
      e2pTR_last->Fill(eTotalWithEcalCor/t_p ,1.0);
    else
      e2pHE_last->Fill(eTotalWithEcalCor/t_p ,1.0);
      
  } // ------------------- end of loop over events -------------------------------------

  if ( isTest ) {
    double fac[N_DEPTHS][MAXNUM_SUBDET]; 
    double dfac[N_DEPTHS][MAXNUM_SUBDET];
    double ieta[N_DEPTHS][MAXNUM_SUBDET];
    double dieta[N_DEPTHS][MAXNUM_SUBDET];
    unsigned int kdep[N_DEPTHS];
    for ( unsigned ik = 0; ik < N_DEPTHS; ik++ ) { kdep[ik] = 0; }
  
    std::map<unsigned int, double>::iterator factorsItr = factors.begin();
    for (factorsItr=factors.begin(); factorsItr != factors.end(); factorsItr++){

      unsigned int detId = factorsItr->first;
      int zside = (detId&ZSIDE_MASK) ? 1 : -1;
      int depth = ((detId>>DEPTH_OFFSET)&DEPTH_MASK) + int(MERGE_PHI_AND_DEPTHS);

      unsigned int kcur = kdep[depth-1];
      ieta[depth-1][kcur] = int((detId>>ETA_OFFSET) & ETA_MASK)*zside;
      dieta[depth-1][kcur] = 0;
      fac[depth-1][kcur] = factorsItr->second;
      dfac[depth-1][kcur] = 0;
      kdep[depth-1]++;
    }
    for ( unsigned ik = 0; ik < N_DEPTHS; ik++ ) {
      double x[MAXNUM_SUBDET], dx[MAXNUM_SUBDET], y[MAXNUM_SUBDET], dy[MAXNUM_SUBDET];
      for ( unsigned im = 0; im < MAXNUM_SUBDET; im++ ) {
	x[im] = ieta[ik][im];
	dx[im] = dieta[ik][im];
	y[im] = fac[ik][im];
	dy[im] = dfac[ik][im];
      }
      TGraphErrors*  g_fac = new TGraphErrors(kdep[ik], x, y, dx, dy);
      sprintf(name, "Applied correction factors, depth %1d", ik+1);
      g_fac->SetTitle(name);
      g_fac->GetXaxis()->SetTitle("i#eta");
      sprintf(name, "Cfacs_depth%1d", ik+1);
      foutRootFile->WriteTObject(g_fac, name);
    }
  }
  //--- fit response distributions ---------------------------------

  double xl = e2p_last->GetMean() - FIT_RMS_INTERVAL*e2p_last->GetRMS();
  double xr = e2p_last->GetMean() + FIT_RMS_INTERVAL*e2p_last->GetRMS();
  e2p_last->Fit("f1","QN", "R", xl, xr);
  xl = f1->GetParameter(1) - FIT_RMS_INTERVAL*f1->GetParameter(2);
  xr = f1->GetParameter(1) + FIT_RMS_INTERVAL*f1->GetParameter(2);
  e2p_last->Fit("f1","QN", "R", xl, xr);

  double fitMPV = f1->GetParameter(1);
  /*
  xl = e2pHB_last->GetMean() - FIT_RMS_INTERVAL*e2pHB_last->GetRMS();
  xr = e2pHB_last->GetMean() + FIT_RMS_INTERVAL*e2pHB_last->GetRMS();
  e2pHB_last->Fit("f1","QN", "R", xl, xr);

  xl = e2pHE_last->GetMean() - FIT_RMS_INTERVAL*e2pHE_last->GetRMS();
  xr = e2pHE_last->GetMean() + FIT_RMS_INTERVAL*e2pHE_last->GetRMS();
  e2pHE_last->Fit("f1","QN", "R", xl, xr);
  */
  return fitMPV;
}

//**********************************************************
// Isolated track selection
//**********************************************************
Bool_t CalibTree::goodTrack(int ieta) 
{

  double maxCharIso = limCharIso*exp(abs(ieta)*constForFlexSel);

  bool ok = (    (t_selectTk)
	      && (t_qltyMissFlag)
	      && (t_hmaxNearP < maxCharIso)
	      && (t_eMipDR < limMipEcal) 
	      && (t_p > minTrackMom) && (t_p < maxTrackMom)
	      && (t_pt >= minTrackPt)               // constraint on track pt
	      && (t_eHcal >= minEnrHcal)            // constraint on Hcal energy
	      && (t_eHcal/t_p < UPPER_LIMIT_RESPONSE_BEFORE_COR)
		 // reject events with too big cluster energy
		 //&& ((t_eHcal30 - t_eHcal10)/t_p < UPPER_LIMIT_DELTA_PU_COR)
		 // reject events with too high PU in the ring around cluster
	     );
  return ok;
}
//**********************************************************
// Save txt file with calculated factors
//**********************************************************
unsigned int CalibTree::saveFactorsInFile(std::string txtFileName)
{
  char sprnt[100];
  
  FILE* foutTxtFile = fopen(txtFileName.c_str(),"w+");
  fprintf(foutTxtFile,
	  "%1s%16s%16s%16s%9s%11s\n","#", "eta", "depth", "det", "value", "DetId");

  std::cout << "New factors:" << std::endl;
  std::map<unsigned int, double>::iterator factorsItr = factors.begin();
  unsigned int indx(0);
  unsigned int isave(0);
  
  for (factorsItr=factors.begin(); factorsItr != factors.end(); factorsItr++, indx++){
    unsigned int detId = factorsItr->first;
    int ieta = (detId>>ETA_OFFSET) & ETA_MASK;
    int zside= (detId&ZSIDE_MASK) ? 1 : -1;
    int depth= ((detId>>DEPTH_OFFSET)&DEPTH_MASK) + int(MERGE_PHI_AND_DEPTHS);

    double erWeight = 100*uncFromWeights[detId]/factorsItr->second;
    double erDev = 100*uncFromDeviation[detId]/factorsItr->second;
    double erTotal = 100*sqrt(pow(uncFromWeights[detId],2)
			  + pow(uncFromDeviation[detId],2))/factorsItr->second;
    
    if ( N_ETA_RINGS_PER_BIN < 2 ) { 
      sprintf(sprnt,
	      "DetId[%3d] %x (%3d,%1d)  %6.4f  : %6d  [%8.3f%% + %8.3f%% = %8.3f%%]",
	      indx, detId, ieta*zside, depth,
	      factorsItr->second, nTrks[detId],
	      erWeight, erDev, erTotal);
      std::cout << sprnt << std::endl;
    }
    else {
      int ieta_min = ieta - (N_ETA_RINGS_PER_BIN - 1);
      sprintf(sprnt,
	      "DetId[%3d] %x (%3d:%3d,%1d)  %6.4f  : %6d  [%8.3f%% + %8.3f%% = %8.3f%%]",
	      indx, detId, ieta_min*zside, ieta*zside, depth,
	      factorsItr->second, nTrks[detId],
	      erWeight, erDev, erTotal);
      std::cout << sprnt << std::endl;
    }
	    /*
    std::cout << "DetId[" << indx << "] " << std::hex  << (detId) << std::dec 
	      << "(" << ieta*zside << "," << depth << ") ( nTrks:" 
	      << nTrks[detId] << ") : " << factorsItr->second
	      << ""
	      << std::endl;
	    */
    
    const char* subDetector[2] = {"HB","HE"};
    if ( nTrks[detId] < MIN_N_TRACKS_PER_CELL ) continue;
    isave++;
    fprintf(foutTxtFile, "%17i%16i%16s%9.5f%11X\n", 
	    ieta*zside, depth, subDetector[subDetector_final[detId]-1],
	    factorsItr->second, detId);
  }
  fclose(foutTxtFile);
  foutTxtFile = NULL;
  return isave;
}
//**********************************************************
// Get factors from txt file
//**********************************************************
Bool_t CalibTree::getFactorsFromFile(std::string txtFileName,
				     unsigned int dbg)
{

  if ( !gSystem->Which("./", txtFileName.c_str() ) ) return false;

  FILE* finTxtFile = fopen(txtFileName.c_str(),"r");
  int flag;

  char header[80]; 
  for ( unsigned int i = 0; i < 6; i++ ) { 
    flag = fscanf(finTxtFile, "%7s", header);
  }

  int eta;
  int depth;
  char det[2]; 
  double cellFactor;
  unsigned int detId;
  unsigned int nReadFactors(0);
  
  while ( fscanf(finTxtFile, "%3d", &eta) != EOF )
    {
      flag = fscanf(finTxtFile, "%2d", &depth);
      flag = fscanf(finTxtFile, "%10s", det);
      flag = fscanf(finTxtFile, "%lf", &cellFactor);
      flag = fscanf(finTxtFile, "%x", &detId);
      factors.insert( std::pair<unsigned int, double>(detId, cellFactor) );
      nReadFactors++;
      if ( dbg > 0 ) 
	std::cout << "  " << std::dec << cellFactor
		  << "  " << std::hex << detId << std::endl; 
    }

  std::cout << std::dec << nReadFactors << " factors read from file "
	    << txtFileName
	    << std::endl;
  
  return true;
}
//**********************************************************
