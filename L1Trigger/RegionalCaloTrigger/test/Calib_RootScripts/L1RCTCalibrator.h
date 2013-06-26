//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Mon Sep 29 12:19:26 2008 by ROOT version 5.18/00a
// from TTree L1RCTCalibrator/RCT Calibration Tree
// found on file: /scratch/lgray/PGun2pi2gamma_calibration/rctCalibratorFarmout_cfg-merge_cfg-PGun2pi2gamma-0034.root
//////////////////////////////////////////////////////////

#ifndef L1RCTCalibrator_h
#define L1RCTCalibrator_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH2F.h>
#include <TGraphAsymmErrors.h>
#include <TGraph2DErrors.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <TMath.h>
#include <TF1.h>
#include <TF2.h>

class generator
{
 public:
  int particle_type;
  double et, phi, eta;
  
  bool operator==(const generator& r) const { return ( particle_type == r.particle_type && et == r.et &&
						       phi == r.phi && eta == r.eta ); }
};

class region
{
 public:
  int ieta, iphi, linear_et;
  double eta,phi;

  bool operator==(const region& r) const { return ((ieta == r.ieta) && (iphi == r.iphi)); }
};

class tpg
{
 public:    
  int ieta, iphi;
  double ecalEt, hcalEt, ecalE, hcalE;
  double eta,phi;
  
  bool operator==(const tpg& r) const { return ((ieta == r.ieta) && (iphi == r.iphi)); }
};

class event_data
{
 public:
  int event, run;
  std::vector<generator> gen_particles;
  std::vector<region> regions;
  std::vector<tpg> tpgs;
};

class L1RCTCalibrator {
public :

  typedef TH1F* TH1Fptr;
  typedef TH2F* TH2Fptr;
  typedef TGraphAsymmErrors* TGraphptr;
  typedef TGraph2DErrors* TGraph2Dptr;

   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

   // Declaration of leaf types
   UInt_t          Event_event;
   UInt_t          Event_run;
   UInt_t          Generator_nGen;
   Int_t           Generator_particle_type[100];
   Double_t        Generator_et[100];
   Double_t        Generator_eta[100];
   Double_t        Generator_phi[100];
   UInt_t          Generator_crate[100];
   UInt_t          Generator_card[100];
   UInt_t          Generator_region[100];
   UInt_t          Region_nRegions;
   Int_t           Region_linear_et[200];
   Int_t           Region_ieta[200];
   Int_t           Region_iphi[200];
   Double_t        Region_eta[200];
   Double_t        Region_phi[200];
   UInt_t          Region_crate[200];
   UInt_t          Region_card[200];
   UInt_t          Region_region[200];
   UInt_t          CaloTPG_nTPG;
   Int_t           CaloTPG_ieta[3100];
   Int_t           CaloTPG_iphi[3100];
   Double_t        CaloTPG_eta[3100];
   Double_t        CaloTPG_phi[3100];
   Double_t        CaloTPG_ecalEt[3100];
   Double_t        CaloTPG_hcalEt[3100];
   Double_t        CaloTPG_ecalE[3100];
   Double_t        CaloTPG_hcalE[3100];
   UInt_t          CaloTPG_crate[3100];
   UInt_t          CaloTPG_card[3100];
   UInt_t          CaloTPG_region[3100];

   // List of branches
   TBranch        *b_Event;   //!
   TBranch        *b_Generator;   //!
   TBranch        *b_Region;   //!
   TBranch        *b_CaloTPG;   //!

   L1RCTCalibrator(TTree *tree=0);
   virtual ~L1RCTCalibrator();
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
   void BookHistos();
   void WriteHistos();

   // ----------protected member functions---------------
  void deltaR(const double& eta1, double phi1, 
	      const double& eta2, double phi2,double& dr) const; // calculates delta R between two coordinates
  void etaBin(const double&, int&) const; // calculates Trigger Tower number
  void etaValue(const int&, double&) const; // calculates Trigger Tower eta bin center
  void phiBin(double, int&) const; // calculates TT phi bin
  void phiValue(const int&, double&) const; // calculates TT phi bin center
  double uniPhi(const double&) const; // returns phi that is in [0, 2*pi]

  // returns -1 if item not present, [index] if it is, T must have an == operator
  template<typename T> 
    int find(const T&, const std::vector<T>&) const;

  // returns tpgs within a specified delta r near the point (eta,phi)
  std::vector<tpg> tpgsNear(const double& eta, const double& phi, const std::vector<tpg>&, const double& dr = .5) const;
  // returns an ordered pair of (Et, deltaR)
  std::pair<double,double> showerSize(const std::vector<tpg>&, const double frac = .95, const double& max_dr = .5, 
				      const bool& ecal = true, const bool& hcal = true) const;
  // returns the sum of tpg Et near the point (eta,phi) within a specified delta R, 
  // can choose to only give ecal or hcal sum through bools
  double sumEt(const double& eta, const double& phi, const std::vector<region>&, const double& dr = .5) const;
  double sumEt(const double& eta, const double& phi, const std::vector<tpg>&, const double& dr = .5, 
	       const bool& ecal = true, const bool& hcal = true, const bool& apply_corrections = false,
	       const double& high_low_crossover = 23) const;
  // returns energy weighted average of Eta
  double avgPhi(const std::vector<tpg>&) const;
  // returns energy weighted average of Phi
  double avgEta(const std::vector<tpg>&) const;


   bool sanityCheck() const;
   // prints a CFG or python language configuration file fragment that contains the
  // resultant calibration information
  void printCfFragment(std::ostream&) const;

  std::vector<TObject*> hists_;

  // saves pointer to Histogram in a vector, making writing out easier later.
  void putHist(TObject* o) { hists_.push_back(o); }

  //do the calibration
  void makeCalibration();
  
  //finds overlapping particles
  std::vector<generator> overlaps(const std::vector<generator>& v) const;

  bool python_;
  const std::string& fitOpts() const { return fitOpts_; }
  const int& debug() const { return debug_; }

  int debug_;
  std::string fitOpts_;
  TFile* output_;
  std::vector<event_data> data_;

  const double deltaEtaBarrel_, maxEtaBarrel_, deltaPhi_;
  std::vector<double> endcapEta_;

  //The final output... correction factors
  double ecal_[28][3], hcal_[28][3], hcal_high_[28][3], cross_[28][6], he_low_smear_[28], he_high_smear_[28];

  // histograms

  // diagnostic histograms
  TH1Fptr hEvent, hRun, hGenPhi, hGenEta, hGenEt, hGenEtSel, 
    hRCTRegionEt, hRCTRegionPhi, hRCTRegionEta,
    hTpgSumEt, hTpgSumEta, hTpgSumPhi;
  
  TH2Fptr hGenPhivsTpgSumPhi, hGenEtavsTpgSumEta, hGenPhivsRegionPhi, hGenEtavsRegionEta;

  TH1Fptr hDeltaEtPeakvsEtaBin_uc[12], hDeltaEtPeakvsEtaBin_c[12], hDeltaEtPeakRatiovsEtaBin[12], 
    hPhotonDeltaEtPeakvsEtaBinAllEt_uc, hPhotonDeltaEtPeakvsEtaBinAllEt_c, hPhotonDeltaEtPeakRatiovsEtaBinAllEt,
    hPionDeltaEtPeakvsEtaBinAllEt_uc, hPionDeltaEtPeakvsEtaBinAllEt_c, hPionDeltaEtPeakRatiovsEtaBinAllEt;

  TH1Fptr hPhotonDeltaR95[28], hNIPionDeltaR95[28], hPionDeltaR95[28] ;
  
  // histograms for algorithm
  TGraphptr gPhotonEtvsGenEt[28], gNIPionEtvsGenEt[28];
  TGraph2Dptr gPionEcalEtvsHcalEtvsGenEt[28];

  TH1Fptr hPhotonDeltaEOverE_uncor[28], hPionDeltaEOverE_uncor[28];
  TH1Fptr hPhotonDeltaEOverE[28], hPionDeltaEOverE[28];
  TH1Fptr hPhotonDeltaEOverE_cor[28], hPionDeltaEOverE_cor[28];
};

#endif 

#ifdef L1RCTCalibrator_cxx
L1RCTCalibrator::L1RCTCalibrator(TTree *tree):deltaEtaBarrel_(0.0870), maxEtaBarrel_(20*deltaEtaBarrel_),deltaPhi_(0.0870) 
{
  python_ = true;
  debug_ = 0;
  fitOpts_ = ((debug_ > -1) ? "ELMRNF" : "QELMRNF");
  output_ = new TFile("L1RCTCalibration_info.root","RECREATE");
    
  endcapEta_.push_back(0.09);
  endcapEta_.push_back(0.1);
  endcapEta_.push_back(0.113);
  endcapEta_.push_back(0.129);
  endcapEta_.push_back(0.15);
  endcapEta_.push_back(0.178);
  endcapEta_.push_back(0.15);
  endcapEta_.push_back(0.35);
  
  for(int i = 0; i < 28; ++i)
    {
      he_low_smear_[i]  = -999;
      he_high_smear_[i] = -999;
      for(int j = 0; j < 6; ++j)
        {
          if(j < 3)
            {
              ecal_[i][j]      = -999;
              hcal_[i][j]      = -999;
              hcal_high_[i][j] = -999;
            }
          cross_[i][j] = -999;
        }
    }


// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("/scratch/lgray/PGun2pi2gamma_calibration/rctCalibratorFarmout_cfg-merge_cfg-PGun2pi2gamma-0034.root");
      if (!f) {
         f = new TFile("/scratch/lgray/PGun2pi2gamma_calibration/rctCalibratorFarmout_cfg-merge_cfg-PGun2pi2gamma-0034.root");
      }
      tree = (TTree*)gDirectory->Get("L1RCTCalibrator");

   }
   Init(tree);
   BookHistos();
}

L1RCTCalibrator::~L1RCTCalibrator()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t L1RCTCalibrator::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t L1RCTCalibrator::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (!fChain->InheritsFrom(TChain::Class()))  return centry;
   TChain *chain = (TChain*)fChain;
   if (chain->GetTreeNumber() != fCurrent) {
      fCurrent = chain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void L1RCTCalibrator::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("Event", &Event_event, &b_Event);
   fChain->SetBranchAddress("Generator", &Generator_nGen, &b_Generator);
   fChain->SetBranchAddress("Region", &Region_nRegions, &b_Region);
   fChain->SetBranchAddress("CaloTPG", &CaloTPG_nTPG, &b_CaloTPG);
   Notify();
}

Bool_t L1RCTCalibrator::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void L1RCTCalibrator::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}

template<typename T> 
int L1RCTCalibrator::find(const T& item, const std::vector<T>& v) const
{
  for(unsigned i = 0; i < v.size(); ++i)
    if(item == v[i]) return i;
  return -1;
}

#endif // #ifdef L1RCTCalibrator_cxx
