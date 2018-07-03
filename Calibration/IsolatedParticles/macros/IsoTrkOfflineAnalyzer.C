//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Jul 30 20:56:15 2015 by ROOT version 5.26/00
// from TTree CalibTree/CalibTree
// found on file: output_all.root
//////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TProfile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

class IsoTrkOfflineAnalyzer {
public :
  TTree          *fChain;   //!pointer to the analyzed TTree or TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t                      t_Run;
  Int_t                      t_Event;
  Int_t                      t_ieta;
  Int_t                      t_goodPV;
  Double_t                   t_EventWeight;
  Double_t                   t_l1pt;
  Double_t                   t_l1eta;
  Double_t                   t_l1phi;
  Double_t                   t_l3pt;
  Double_t                   t_l3eta;
  Double_t                   t_l3phi;
  Double_t                   t_p;
  Double_t                   t_mindR1;
  Double_t                   t_mindR2;
  Double_t                   t_eMipDR;
  Double_t                   t_eHcal;
  Double_t                   t_hmaxNearP;
  Bool_t                     t_selectTk;
  Bool_t                     t_qltyFlag;
  Bool_t                     t_qltyMissFlag;
  Bool_t                     t_qltyPVFlag;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double>       *t_HitEnergies;
  std::vector<bool>         *t_trgbits;

  // List of branches
  TBranch                   *b_t_Run;           //!
  TBranch                   *b_t_Event;         //!
  TBranch                   *b_t_ieta;          //!
  TBranch                   *b_t_goodPV;        //!
  TBranch                   *b_t_EventWeight;   //!
  TBranch                   *b_t_l1pt;          //!
  TBranch                   *b_t_l1eta;         //!
  TBranch                   *b_t_l1phi;         //!
  TBranch                   *b_t_l3pt;          //!
  TBranch                   *b_t_l3eta;         //!
  TBranch                   *b_t_l3phi;         //!
  TBranch                   *b_t_p;             //!
  TBranch                   *b_t_mindR1;        //!
  TBranch                   *b_t_mindR2;        //!
  TBranch                   *b_t_eMipDR;        //!
  TBranch                   *b_t_eHcal;         //!
  TBranch                   *b_t_hmaxNearP;     //!
  TBranch                   *b_t_selectTk;      //!
  TBranch                   *b_t_qltyFlag;      //!
  TBranch                   *b_t_qltyMissFlag;  //!
  TBranch                   *b_t_qltyPVFlag;    //!
  TBranch                   *b_t_DetIds;        //!
  TBranch                   *b_t_HitEnergies;   //!
  TBranch                   *b_t_trgbits;       //!
  struct record {
    record() {
      serial_ = entry_ = run_ = event_ = ieta_ = p_ = 0;
    };
    record(int ser, Long64_t ent, int r, int ev, int ie, double p) :
      serial_(ser), entry_(ent), run_(r), event_(ev), ieta_(ie), p_(p) {};
    int      serial_;
    Long64_t entry_;
    int      run_, event_, ieta_;
    double   p_;
  };

  IsoTrkOfflineAnalyzer(std::string fname, std::string dirname, std::string prefix="", int flag=0, bool datMC=true, std::string listname="runEventFile.txt");
  virtual ~IsoTrkOfflineAnalyzer();
  virtual Int_t              Cut(Long64_t entry);
  virtual Int_t              GetEntry(Long64_t entry);
  virtual Long64_t           LoadTree(Long64_t entry);
  virtual void               Init(TTree *tree);
  virtual void               Loop();
  virtual Bool_t             Notify();
  virtual void               Show(Long64_t entry = -1);
  std::vector<Long64_t>      findDuplicate (std::vector<IsoTrkOfflineAnalyzer::record>&);
  void                       PlotHist(int type, int num, bool save=false);
  template<class Hist> void  DrawHist(Hist*, TCanvas*);
  void                       SavePlot(std::string theName, bool append, bool all=false);
private:

  static const unsigned int npbin=10, kp50=7;
  std::string               fname_, dirnm_, prefix_, listName_;
  int                       flag_;
  bool                      dataMC_, plotStandard_;
  std::vector<double>       etas_, ps_, dl1_;
  std::vector<int>          nvx_;
  TH1D                     *h_p[5], *h_eta[5];
  std::vector<TH1D*>        h_eta0, h_eta1, h_eta2, h_eta3, h_eta4;
  std::vector<TH1D*>        h_dL1,  h_vtx, h_etaF[npbin];
  std::vector<TProfile*>    h_etaX[npbin];
  std::vector<TH1D*>        h_etaR[npbin], h_nvxR[npbin], h_dL1R[npbin];
};

IsoTrkOfflineAnalyzer::IsoTrkOfflineAnalyzer(std::string fname, std::string dirnm, std::string prefix, int flag, bool datMC, std::string listname) : fname_(fname), dirnm_(dirnm), prefix_(prefix), listName_(listname), flag_(flag), dataMC_(datMC) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree
  plotStandard_    = (((flag_/10000)%10) == 0);
  TFile      *file = new TFile(fname.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
  std::cout << fname << " file " << file << " " << dirnm << " " << dir << std::endl;
  TTree      *tree = (TTree*)dir->Get("CalibTree");
  std::cout << "CalibTree " << tree << std::endl;
  Init(tree);
}

IsoTrkOfflineAnalyzer::~IsoTrkOfflineAnalyzer() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t IsoTrkOfflineAnalyzer::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t IsoTrkOfflineAnalyzer::LoadTree(Long64_t entry) {
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

void IsoTrkOfflineAnalyzer::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).
  
  // Set object pointer
  t_DetIds      = 0;
  t_HitEnergies = 0;
  t_trgbits     = 0;
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_goodPV", &t_goodPV, &b_t_goodPV);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_l1pt", &t_l1pt, &b_t_l1pt);
  fChain->SetBranchAddress("t_l1eta", &t_l1eta, &b_t_l1eta);
  fChain->SetBranchAddress("t_l1phi", &t_l1phi, &b_t_l1phi);
  fChain->SetBranchAddress("t_l3pt", &t_l3pt, &b_t_l3pt);
  fChain->SetBranchAddress("t_l3eta", &t_l3eta, &b_t_l3eta);
  fChain->SetBranchAddress("t_l3phi", &t_l3phi, &b_t_l3phi);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyFlag", &t_qltyFlag, &b_t_qltyFlag);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  fChain->SetBranchAddress("t_trgbits", &t_trgbits, &b_t_trgbits);
  Notify();

  double xbins[9] = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
  double xbina[43]= {-21.5,-20.5,-19.5,-18.5,-17.5,-16.5,-15.5,-14.5,-13.5,
		     -12.5,-11.5,-10.5,-9.5,-8.5,-7.5,-6.5,-5.5,-4.5,-3.5,
		     -2.5,-1.5,0.0,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,
		     11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5};
  if (plotStandard_) {
    for (int i=0; i<9; ++i) etas_.push_back(xbins[i]);
  } else {
    for (int i=0; i<43; ++i) etas_.push_back(xbina[i]);
  }
  int ipbin[10] = {2, 4, 7, 10, 15, 20, 30, 40, 60, 100};
  for (int i=0; i<10; ++i) ps_.push_back((double)(ipbin[i]));
  int npvtx[6]  = {0, 7,10, 13, 16,100};
  for (int i=0; i<6; ++i)  nvx_.push_back(npvtx[i]);
  double dl1s[9]= {0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};

  char        name[20], title[100];
  std::string titl[5] = {"All tracks", "Good quality tracks", "Selected tracks",
			 "Tracks with charge isolation", "Tracks MIP in ECAL"};
  for (int i=0; i<9; ++i)  dl1_.push_back(dl1s[i]);
  if (plotStandard_) {
    std::cout << "Book Histos for Standard\n";
    for (int k=0; k<5; ++k) {
      sprintf (name, "%sp%d", prefix_.c_str(), k);
      sprintf (title,"%s", titl[k].c_str());
      h_p[k] = new TH1D(name, title, 100, 10.0, 110.0);
      sprintf (name, "%seta%d", prefix_.c_str(), k);
      sprintf (title,"%s", titl[k].c_str());
      h_eta[k] = new TH1D(name, title, 60, -30.0, 30.0);
    }
    unsigned int kp = (ps_.size()-1);
    for (unsigned int k=0; k<kp; ++k) {
      sprintf (name, "%seta0%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[0].c_str(),ipbin[k],ipbin[k+1]);
      h_eta0.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta1%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[1].c_str(),ipbin[k],ipbin[k+1]);
      h_eta1.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta2%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[2].c_str(),ipbin[k],ipbin[k+1]);
      h_eta2.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta3%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[3].c_str(),ipbin[k],ipbin[k+1]);
      h_eta3.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%seta4%d", prefix_.c_str(), k);
      sprintf (title,"%s (p = %d:%d GeV)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
      h_eta4.push_back(new TH1D(name, title, 60, -30.0, 30.0));
      sprintf (name, "%sdl1%d", prefix_.c_str(), k);
      sprintf (title,"Distance from L1 (p = %d:%d GeV)",ipbin[k],ipbin[k+1]);
      h_dL1.push_back(new TH1D(name, title, 160, 0.0, 8.0));
      sprintf (name, "%svtx%d", prefix_.c_str(), k);
      sprintf (title,"N_{Vertex} (p = %d:%d GeV)",ipbin[k],ipbin[k+1]);
      h_vtx.push_back(new TH1D(name, title, 100, 0.0, 100.0));
      for (unsigned int i=0; i<nvx_.size(); ++i) {
	sprintf (name, "%setaX%d%d", prefix_.c_str(), k, i);
	if (i == 0) {
	  sprintf (title,"%s (p = %d:%d GeV all vertices)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"%s (p = %d:%d GeV # Vtx %d:%d)",titl[4].c_str(),ipbin[k],ipbin[k+1],nvx_[i-1],nvx_[i]);
	}
	h_etaX[k].push_back(new TProfile(name, title, 8, xbins));
	unsigned int kk = h_etaX[k].size()-1;
	h_etaX[k][kk]->Sumw2();
	sprintf (name, "%snvxR%d%d", prefix_.c_str(), k, i);
	if (i == 0) {
	  sprintf (title,"E/p for %s (p = %d:%d GeV all vertices)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"E/p for %s (p = %d:%d GeV # Vtx %d:%d)",titl[4].c_str(),ipbin[k],ipbin[k+1],nvx_[i-1],nvx_[i]);
	}
	h_nvxR[k].push_back(new TH1D(name,title,100,0.,5.));
	kk = h_nvxR[k].size()-1;
	h_nvxR[k][kk]->Sumw2();
      }
      for (unsigned int j=0; j<etas_.size(); ++j) {
	sprintf (name, "%sratio%d%d", prefix_.c_str(), k, j);
	if (j == 0) {
	  sprintf (title,"E/p for %s (p = %d:%d GeV)",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"E/p for %s (p = %d:%d GeV #eta %4.1f:%4.1f)",titl[4].c_str(),ipbin[k],ipbin[k+1],etas_[j-1],etas_[j]);
	}
	h_etaF[k].push_back(new TH1D(name,title,100,0.,5.));
	unsigned int kk = h_etaF[k].size()-1;
	h_etaF[k][kk]->Sumw2();
	sprintf (name, "%setaR%d%d", prefix_.c_str(), k, j);
	h_etaR[k].push_back(new TH1D(name,title,100,0.,5.));
	kk = h_etaR[k].size()-1;
	h_etaR[k][kk]->Sumw2();
      }
      for (unsigned int j=0; j<dl1_.size(); ++j) {
	sprintf (name, "%sdl1R%d%d", prefix_.c_str(), k, j);
	if (j == 0) {
	  sprintf (title,"E/p for %s (p = %d:%d GeV All d_{L1})",titl[4].c_str(),ipbin[k],ipbin[k+1]);
	} else {
	  sprintf (title,"E/p for %s (p = %d:%d GeV d_{L1} %4.2f:%4.2f)",titl[4].c_str(),ipbin[k],ipbin[k+1],dl1_[j-1],dl1_[j]);
	}
	h_dL1R[k].push_back(new TH1D(name,title,100,0.,5.));
	unsigned int kk = h_dL1R[k].size()-1;
	h_dL1R[k][kk]->Sumw2();
      }
    }
    for (unsigned int i=0; i<nvx_.size(); ++i) {
      sprintf (name, "%setaX%d%d", prefix_.c_str(), kp, i);
      if (i == 0) {
	sprintf (title,"%s (All Momentum all vertices)",titl[4].c_str());
      } else {
	sprintf (title,"%s (All Momentum # Vtx %d:%d)",titl[4].c_str(),nvx_[i-1],nvx_[i]);
      }
      h_etaX[npbin-1].push_back(new TProfile(name, title, 8, xbins));
      unsigned int kk = h_etaX[npbin-1].size()-1;
      h_etaX[npbin-1][kk]->Sumw2();
      sprintf (name, "%snvxR%d%d", prefix_.c_str(), kp, i);
      if (i == 0) {
	sprintf (title,"E/p for %s (All Momentum all vertices)",titl[4].c_str());
      } else {
	sprintf (title,"E/p for %s (All Momentum # Vtx %d:%d)",titl[4].c_str(),nvx_[i-1],nvx_[i]);
      }
      h_nvxR[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      kk = h_nvxR[npbin-1].size()-1;
      h_nvxR[npbin-1][kk]->Sumw2();
    }
    for (unsigned int j=0; j<etas_.size(); ++j) {
      sprintf (name, "%sratio%d%d", prefix_.c_str(), kp, j);
      if (j == 0) {
	sprintf (title,"E/p for %s (All momentum)",titl[4].c_str());
      } else {
	sprintf (title,"E/p for %s (All momentum #eta %4.1f:%4.1f)",titl[4].c_str(),etas_[j-1],etas_[j]);
      }
      h_etaF[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      unsigned int kk = h_etaF[npbin-1].size()-1;
      h_etaF[npbin-1][kk]->Sumw2();
      sprintf (name, "%setaR%d%d", prefix_.c_str(), kp, j);
      h_etaR[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      kk = h_etaR[npbin-1].size()-1;
      h_etaR[npbin-1][kk]->Sumw2();
    }
    for (unsigned int j=0; j<dl1_.size(); ++j) {
      sprintf (name, "%sdl1R%d%d", prefix_.c_str(), kp, j);
      if (j == 0) {
	sprintf (title,"E/p for %s (All momentum)",titl[4].c_str());
      } else {
	sprintf (title,"E/p for %s (All momentum d_{L1} %4.2f:%4.2f)",titl[4].c_str(),dl1_[j-1],dl1_[j]);
      }
      h_dL1R[npbin-1].push_back(new TH1D(name,title,200,0.,10.));
      unsigned int kk = h_dL1R[npbin-1].size()-1;
      h_dL1R[npbin-1][kk]->Sumw2();
    }
  } else {
    std::cout << "Book Histos for Non-Standard " << etas_.size() << ":" << kp50 << "\n";
    for (unsigned int j=0; j<etas_.size(); ++j) {
      sprintf (name, "%sratio%d%d", prefix_.c_str(), kp50, j);
      if (j == 0) {
	sprintf (title,"E/p for %s (p = %d:%d GeV)",titl[4].c_str(),ipbin[kp50],ipbin[kp50+1]);
      } else {
	sprintf (title,"E/p for %s (p = %d:%d GeV #eta %4.1f:%4.1f)",titl[4].c_str(),ipbin[kp50],ipbin[kp50+1],etas_[j-1],etas_[j]);
      }
      h_etaF[kp50].push_back(new TH1D(name,title,100,0.,5.));
      unsigned int kk = h_etaF[kp50].size()-1;
      h_etaF[kp50][kk]->Sumw2();
    }
  }
}

Bool_t IsoTrkOfflineAnalyzer::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void IsoTrkOfflineAnalyzer::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t IsoTrkOfflineAnalyzer::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void IsoTrkOfflineAnalyzer::Loop() {
  //   In a ROOT session, you can do:
  //      Root > .L IsoTrkOfflineAnalyzer.C
  //      Root > IsoTrkOfflineAnalyzer t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //

  //   This is the loop skeleton where:
  //      jentry is the global entry number in the chain
  //      ientry is the entry number in the current Tree
  //   Note that the argument to GetEntry must be:
  //      jentry for TChain::GetEntry
  //      ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch
  if (fChain == 0) return;

  // Get list of run, event #'s if flag_>0
  std::vector<int> runs, events;
  runs.clear(); events.clear(); 
  std::vector<double> p_s;
  if ((flag_%10)>0) {
    ifstream infile(listName_.c_str());
    if (!infile.is_open()) {
      std::cout << "Cannot open " << listName_ << std::endl;
    } else {
      while (1) {
	int run, event;
	infile >> run >> event;
	if (!infile.good()) break;
	runs.push_back(run); events.push_back(event);
      }
      infile.close();
      std::cout << "Reads a list of " << runs.size() << " events from " << listName_ << std::endl;
      for (unsigned int k=0; k<runs.size(); ++k) std::cout << "[" << k << "] " << runs[k] << " " << events[k] << std::endl;
    }
  }
  std::ofstream file;
  if (((flag_/10)%10)>0) {
    file.open(listName_.c_str(), std::ofstream::out);
    std::cout << "Opens " << listName_ << " file in output mode\n";
  }
  std::ofstream fileout;
  if (((flag_/1000)%10)>0) {
    if (((flag_/1000)%10)==2) {
      fileout.open("events.txt", std::ofstream::out);
      std::cout << "Opens events.txt in output mode\n";
    } else {
      fileout.open("events.txt", std::ofstream::app);
      std::cout << "Opens events.txt in append mode\n";
    }
    fileout << "Input file: " << fname_ << " Directory: " << dirnm_ 
	    << " Prefix: " << prefix_ << "\n";
  }

  // Find list of duplicate events  
  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "Total entries " << nentries << std::endl;
  Long64_t nbytes(0), nb(0);
  std::vector<Long64_t> entries;
  if (((flag_/100)%10)==0) {
    std::vector<IsoTrkOfflineAnalyzer::record> records;
    int                                        kount(0);
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;
      double cut = (t_p > 20) ? 2.0 : 0.0;
      double rcut= (t_p > 20) ? 0.25: 0.1;
      if (t_qltyFlag) {
	if (t_selectTk) {
	  if (t_hmaxNearP < cut) {
	    if (t_eMipDR < 1.0) {
	      double rat = (t_p > 0) ? (t_eHcal/(t_p-t_eMipDR)) : 1.0;
	      if (rat > rcut) {
		kount++;
		IsoTrkOfflineAnalyzer::record rec(kount,jentry,t_Run,t_Event,t_ieta,t_p);
		records.push_back(rec);
	      }
	    }
	  }
	}
      }
    }
    entries = findDuplicate (records);
  } else if (((flag_/100)%10)==1) {
    char filename[100];
    sprintf (filename, "events_%s.txt", prefix_.c_str());
    ifstream infile(filename);
    if (!infile.is_open()) {
      std::cout << "Cannot open " << filename << std::endl;
    } else {
      while (1) {
	Long64_t jentry;
	infile >> jentry;
	if (!infile.good()) break;
	entries.push_back(jentry);
      }
      infile.close();
      std::cout << "Reads a list of " << entries.size() << " events from " 
		<< filename << std::endl;
    }
  }

  nbytes = nb = 0;
  unsigned int  kount(0), duplicate(0), good(0);
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    bool select(true);
    if (runs.size() > 0) {
      select = false;
      for (unsigned int k=0; k<runs.size(); ++k) {
	if (t_Run == runs[k] && t_Event == events[k]) {
	  select = true;
	  break;
	}
      }
      if (!select) std::cout << "Unwanted event " << t_Run << " " << t_Event 
			     << std::endl;
    }
    if ((entries.size() > 0) && (((flag_/100)%10)<=1)) {
      for (unsigned int k=0; k<entries.size(); ++k) {
	if (jentry == entries[k]) {
//	  std::cout << "Duplicate event " << t_Run << " " << t_Event << " " << t_p << std::endl;
	  select = false;
	  duplicate++;
	  break;
	}
      }
    }
    if (((flag_/10)%10)>0) {
      if (t_p >= 20.0) {
	file << t_Run << " " << t_Event << "\n";
	++kount;
      }
    }
    if (!select) continue;

    // if (Cut(ientry) < 0) continue;
    int kp(-1), jp(-1);
    for (unsigned int k=1; k<ps_.size(); ++k ) {
      if (t_p >= ps_[k-1] && t_p < ps_[k]) {
	kp = k - 1; break;
      }
    }
    unsigned int kp1 = ps_.size() - 1;
    unsigned int kv1 = 0;
    unsigned int kv  = nvx_.size() - 1;
    for (unsigned int k=1; k<nvx_.size(); ++k ) {
      if (t_goodPV >= nvx_[k-1] && t_goodPV < nvx_[k]) {
	kv = k; break;
      }
    }
    unsigned int kd1 =  0;
    unsigned int kd  = dl1_.size() - 1;
    for (unsigned int k=1; k<dl1_.size(); ++k ) {
      if (t_mindR1 >= dl1_[k-1] && t_mindR1 < dl1_[k]) {
	kd = k; break;
      }
    }
    double eta = (t_ieta > 0) ? ((double)(t_ieta)-0.001) : ((double)(t_ieta)+0.001);
    for (unsigned int j=1; j<etas_.size(); ++j) {
      if (eta > etas_[j-1] && eta < etas_[j]) {
	jp = j; break;
      }
    }
//    std::cout << "Bin " << kp << ":" << kp1 << ":" << kv << ":" << kv1 << ":" << kd << ":" << kd1 << ":" << jp << std::endl;
    if (plotStandard_) {
      h_p[0]->Fill(t_p,t_EventWeight);
      h_eta[0]->Fill(t_ieta,t_EventWeight);
      if (kp >= 0) h_eta0[kp]->Fill(t_ieta,t_EventWeight);
    }
    double rat = (t_p > 0) ? (t_eHcal/(t_p-t_eMipDR)) : 1.0;
    double cut = (t_p > 20) ? 2.0 : 0.0;
    double rcut= (t_p > 20) ? 0.25: 0.1;
//    std::cout << "Entry " << jentry << " p|ratio " << t_p << "|" << rat << "|" << kp << "|" << kv << "|" << jp << " Cuts " << t_qltyFlag << "|" << t_selectTk << "|" << (t_hmaxNearP < cut) << "|" << (t_eMipDR < 1.0) << std::endl;
    if (t_qltyFlag) {
      if (plotStandard_) {
	h_p[1]->Fill(t_p,t_EventWeight);
	h_eta[1]->Fill(t_ieta,t_EventWeight);
	if (kp >= 0) h_eta1[kp]->Fill(t_ieta,t_EventWeight);
      }
      if (t_selectTk) {
	if (plotStandard_) {
	  h_p[2]->Fill(t_p,t_EventWeight);
	  h_eta[2]->Fill(t_ieta,t_EventWeight);
	  if (kp >= 0) h_eta2[kp]->Fill(t_ieta,t_EventWeight);
	}
	if (t_hmaxNearP < cut) {
	  if (plotStandard_) {
	    h_p[3]->Fill(t_p,t_EventWeight);
	    h_eta[3]->Fill(t_ieta,t_EventWeight);
	    if (kp >= 0) h_eta3[kp]->Fill(t_ieta,t_EventWeight);
	  }
	  if (t_eMipDR < 1.0) {
	    if (plotStandard_) {
	      h_p[4]->Fill(t_p,t_EventWeight);
	      h_eta[4]->Fill(t_ieta,t_EventWeight);
	    }
	    if (kp >= 0) {
	      if (plotStandard_) {
		h_eta4[kp]->Fill(t_ieta,t_EventWeight);
		h_dL1[kp]->Fill(t_mindR1,t_EventWeight);
		h_vtx[kp]->Fill(t_goodPV,t_EventWeight);
	      }
	      if (rat > rcut) {
		if (plotStandard_) {
		  h_etaX[kp][kv]->Fill(eta,rat,t_EventWeight);
		  h_etaX[kp][kv1]->Fill(eta,rat,t_EventWeight);
		  h_nvxR[kp][kv]->Fill(rat,t_EventWeight);
		  h_nvxR[kp][kv1]->Fill(rat,t_EventWeight);
		  h_dL1R[kp][kd]->Fill(rat,t_EventWeight);
		  h_dL1R[kp][kd1]->Fill(rat,t_EventWeight);
		  if (jp > 0) h_etaR[kp][jp]->Fill(rat,t_EventWeight);
		  h_etaR[kp][0]->Fill(rat,t_EventWeight);
		}
		if ((!dataMC_) || (t_mindR1 > 0.5)) {
		  if (plotStandard_) {
		    if (jp > 0) h_etaF[kp][jp]->Fill(rat,t_EventWeight);
		    h_etaF[kp][0]->Fill(rat,t_EventWeight);
		  }
		}
		if ((!plotStandard_) && (kp == (int)(kp50))) {
//		  std::cout << "kp " << kp << h_etaF[kp].size() << std::endl;
		  if ((!dataMC_) || (t_mindR1>0.5) || (((flag_/10000)%10)==2)) {
		    if (jp > 0) h_etaF[kp][jp]->Fill(rat,t_EventWeight);
		    h_etaF[kp][0]->Fill(rat,t_EventWeight);
		  }
		}
	      }
	    }
	    if (t_p > 20.0 && rat > rcut) {
	      if (plotStandard_) {
		h_etaX[kp1][kv]->Fill(eta,rat,t_EventWeight);
		h_etaX[kp1][kv1]->Fill(eta,rat,t_EventWeight);
		h_nvxR[kp1][kv]->Fill(rat,t_EventWeight);
		h_nvxR[kp1][kv1]->Fill(rat,t_EventWeight);
		h_dL1R[kp1][kd]->Fill(rat,t_EventWeight);
		h_dL1R[kp1][kd1]->Fill(rat,t_EventWeight);
		if (jp > 0) h_etaR[kp1][jp]->Fill(rat,t_EventWeight);
		h_etaR[kp1][0]->Fill(rat,t_EventWeight);
		if ((!dataMC_) || (t_mindR1 > 0.5)) {
		  if (jp > 0) h_etaF[kp1][jp]->Fill(rat,t_EventWeight);
		  h_etaF[kp1][0]->Fill(rat,t_EventWeight);
		}
	      }
	      if (((flag_/1000)%10)!=0) {
		good++;
		fileout << good << " " << jentry << " " << t_Run  << " " 
			<< t_Event << " " << t_ieta << " " << t_p << std::endl;
	      }
	    }
	  }
	}
      }
    }
  }
  if (((flag_/10)%10)>0) {
    file.close();
    std::cout << "Writes " << kount << " records in " << listName_ << std::endl;
  }
  if (((flag_/1000)%10)>0) {
    fileout.close();
    std::cout << "Writes " << good << " events in the file events.txt\n";
  }
  if (((flag_/100)%10)<=1) {
    std::cout << "Finds " << duplicate << " Duplicate events in this file\n";
  }
}

std::vector<Long64_t>  IsoTrkOfflineAnalyzer::findDuplicate (std::vector<IsoTrkOfflineAnalyzer::record>& records){
  // First sort by run number
  for (int c = 0 ; c < ((int)(records.size())-1); c++) {
    for (int d = 0; d < ((int)(records.size())-c-1); d++) {
      if (records[d].run_ > records[d+1].run_) {
        record swap  = records[d];
        records[d]   = records[d+1];
        records[d+1] = swap;
      }
    }
  }
  // Then sort by event number
  for (int c = 0 ; c < ((int)(records.size())-1); c++) {
    for (int d = 0; d < ((int)(records.size())-c-1); d++) {
      if ((records[d].run_ == records[d+1].run_) &&
	  (records[d].event_ > records[d+1].event_)) {
        record swap  = records[d];
        records[d]   = records[d+1];
        records[d+1] = swap;
      }
    }
  }
  // Finally by ieta
  for (int c = 0 ; c < ((int)(records.size())-1); c++) {
    for (int d = 0; d < ((int)(records.size())-c-1); d++) {
      if ((records[d].run_ == records[d+1].run_) &&
	  (records[d].event_ == records[d+1].event_) &&
	  (records[d].ieta_ > records[d+1].ieta_)) {
        record swap  = records[d];
        records[d]   = records[d+1];
        records[d+1] = swap;
      }
    }
  }
  // First sort by run number
  for (int c = 0 ; c < ((int)(records.size())-1); c++) {
    for (int d = 0; d < ((int)(records.size())-c-1); d++) {
      if (records[d].run_ > records[d+1].run_) {
        record swap  = records[d];
        records[d]   = records[d+1];
        records[d+1] = swap;
      }
    }
  }
  // Then sort by event number
  for (int c = 0 ; c < ((int)(records.size())-1); c++) {
    for (int d = 0; d < ((int)(records.size())-c-1); d++) {
      if ((records[d].run_ == records[d+1].run_) &&
	  (records[d].event_ > records[d+1].event_)) {
        record swap  = records[d];
        records[d]   = records[d+1];
        records[d+1] = swap;
      }
    }
  }
  // Finally by ieta
  for (int c = 0 ; c < ((int)(records.size())-1); c++) {
    for (int d = 0; d < ((int)(records.size())-c-1); d++) {
      if ((records[d].run_ == records[d+1].run_) &&
	  (records[d].event_ == records[d+1].event_) &&
	  (records[d].ieta_ > records[d+1].ieta_)) {
        record swap  = records[d];
        records[d]   = records[d+1];
        records[d+1] = swap;
      }
    }
  }
  // Find duplicate events
  std::vector<Long64_t> entries;
  for (unsigned int k=1; k<records.size(); ++k) {
    if ((records[k].run_ == records[k-1].run_) &&
	(records[k].event_ == records[k-1].event_) &&
	(records[k].ieta_ == records[k-1].ieta_) &&
	(fabs(records[k].p_-records[k-1].p_) < 0.0001)) {
      // This is a duplicate event
      entries.push_back(records[k].entry_);
    }
  }
  return entries;
}

void IsoTrkOfflineAnalyzer::PlotHist(int itype, int inum, bool save) {
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);     gStyle->SetOptFit(1);
  char name[100];
  int itmin = (itype >=0 && itype < 14) ? itype : 0;
  int itmax = (itype >=0 && itype < 14) ? itype : 13;
  std::string types[14] = {"p","i#eta","i#eta","i#eta","i#eta","i#eta",
			   "i#eta","i#eta","E_{HCAL}/(p-E_{ECAL})",
			   "E_{HCAL}/(p-E_{ECAL})","E_{HCAL}/(p-E_{ECAL})",
			   "E_{HCAL}/(p-E_{ECAL})","dR_{L1}","Vertex"};
  int nmax[14] = {5, 5, npbin-1, npbin-1, npbin-1, npbin-1, npbin-1, 
		  npbin, npbin, npbin, npbin, npbin, npbin-1,npbin-1};
  for (int type=itmin; type<=itmax; ++type) {
    int inmin = (inum >=0 && inum < nmax[type]) ? inum : 0;
    int inmax = (inum >=0 && inum < nmax[type]) ? inum : nmax[type]-1;
    int kmax  = 1;
    if      (type == 8)  kmax = (int)(etas_.size());
    else if (type == 7)  kmax = (int)(nvx_.size());
    else if (type == 9)  kmax = (int)(etas_.size());
    else if (type == 10) kmax = (int)(nvx_.size());
    else if (type == 11) kmax = (int)(dl1_.size());
    for (int num=inmin; num<=inmax; ++num) {
      for (int k=0; k<kmax; ++k) {
	sprintf (name, "c_%d%d%d", type, num, k);
	TCanvas *pad = new TCanvas(name, name, 700, 500);
	pad->SetRightMargin(0.10);
	pad->SetTopMargin(0.10);
	sprintf (name, "%s", types[type].c_str());
	if (type != 7) {
	  TH1D* hist(0);
	  if      (type == 0)  hist = (TH1D*)(h_p[num]->Clone());
	  else if (type == 1)  hist = (TH1D*)(h_eta[num]->Clone());
	  else if (type == 2)  hist = (TH1D*)(h_eta0[num]->Clone());
	  else if (type == 3)  hist = (TH1D*)(h_eta1[num]->Clone());
	  else if (type == 4)  hist = (TH1D*)(h_eta2[num]->Clone());
	  else if (type == 5)  hist = (TH1D*)(h_eta3[num]->Clone());
	  else if (type == 6)  hist = (TH1D*)(h_eta4[num]->Clone());
	  else if (type == 8)  hist = (TH1D*)(h_etaR[num][k]->Clone());
	  else if (type == 9)  hist = (TH1D*)(h_etaF[num][k]->Clone());
	  else if (type == 10) hist = (TH1D*)(h_nvxR[num][k]->Clone());
	  else if (type == 11) hist = (TH1D*)(h_dL1R[num][k]->Clone());
	  else if (type == 12) hist = (TH1D*)(h_dL1[num]->Clone());
	  else                 hist = (TH1D*)(h_vtx[num]->Clone());
	  hist->GetXaxis()->SetTitle(name);
	  hist->GetYaxis()->SetTitle("Tracks");
	  DrawHist(hist,pad);
	  if (save) {
	    sprintf (name, "c_%s%d%d%d.gif", prefix_.c_str(), type,num,k);
	    pad->Print(name);
	  }
	} else {
	  TProfile* hist = (TProfile*)(h_etaX[num][k]->Clone());
	  hist->GetXaxis()->SetTitle(name);
	  hist->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
	  hist->GetYaxis()->SetRangeUser(0.4,1.6);
	  hist->Fit("pol0","q");
	  DrawHist(hist,pad);
	  if (save) {
	    sprintf (name, "c_%s%d%d%d.gif", prefix_.c_str(), type,num,k);
	    pad->Print(name);
	  }
	}
      }	
    }
  }
}

template<class Hist> void IsoTrkOfflineAnalyzer::DrawHist(Hist* hist, TCanvas* pad) {
  hist->GetYaxis()->SetLabelOffset(0.005);
  hist->GetYaxis()->SetTitleOffset(1.20);
  hist->Draw();
  pad->Update();
  TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
  if (st1 != NULL) {
    st1->SetY1NDC(0.70); st1->SetY2NDC(0.90);
    st1->SetX1NDC(0.55); st1->SetX2NDC(0.90);
  }
  pad->Modified();
  pad->Update();
}

void IsoTrkOfflineAnalyzer::SavePlot(std::string theName, bool append, bool all) {

  TFile* theFile(0);
  if (append) {
    theFile = new TFile(theName.c_str(), "UPDATE");
  } else {
    theFile = new TFile(theName.c_str(), "RECREATE");
  }

  theFile->cd();
  for (unsigned int k=0; k<ps_.size(); ++k) {
    if (plotStandard_) {
      for (unsigned int i=0; i<nvx_.size(); ++i) {
	if (h_etaX[k][i] != 0) {
	  TProfile* hnew = (TProfile*)h_etaX[k][i]->Clone();
	  hnew->Write();
	}
	if (h_nvxR[k].size() > i && h_nvxR[k][i] != 0) {
	  TH1D* hist = (TH1D*)h_nvxR[k][i]->Clone();
	  hist->Write();
	}
      }
    }
    for (unsigned int j=0; j<etas_.size(); ++j) {
      if (plotStandard_ && h_etaR[k][j] != 0) {
	TH1D* hist = (TH1D*)h_etaR[k][j]->Clone();
	hist->Write();
      }
      if (h_etaF[k].size() > j && h_etaF[k][j] != 0) {
	TH1D* hist = (TH1D*)h_etaF[k][j]->Clone();
	hist->Write();
      }
    }
    if (plotStandard_) {
      for (unsigned int j=0; j<dl1_.size(); ++j) {
	if (h_dL1R[k][j] != 0) {
	  TH1D* hist = (TH1D*)h_dL1R[k][j]->Clone();
	  hist->Write();
	}
      }
    }
    if (all && plotStandard_ && ((k+1) < ps_.size())) {
      if (h_eta0[k] != 0) {TH1D* h1 = (TH1D*)h_eta0[k]->Clone(); h1->Write();}
      if (h_eta1[k] != 0) {TH1D* h2 = (TH1D*)h_eta1[k]->Clone(); h2->Write();}
      if (h_eta2[k] != 0) {TH1D* h3 = (TH1D*)h_eta2[k]->Clone(); h3->Write();}
      if (h_eta3[k] != 0) {TH1D* h4 = (TH1D*)h_eta3[k]->Clone(); h4->Write();}
      if (h_eta4[k] != 0) {TH1D* h5 = (TH1D*)h_eta4[k]->Clone(); h5->Write();}
      if (h_dL1[k] != 0)  {TH1D* h6 = (TH1D*)h_dL1[k]->Clone();  h6->Write();}
      if (h_vtx[k] != 0)  {TH1D* h7 = (TH1D*)h_vtx[k]->Clone();  h7->Write();}
    }
  }
  std::cout << "All done\n";
  theFile->Close();
}

class GetEntries {
public :
  TTree                     *fChain;   //!pointer to the analyzed TTree/TChain
  Int_t                      fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t                      t_Tracks;
  Int_t                      t_TracksProp;
  Int_t                      t_TracksSaved;
  std::vector<int>          *t_ietaAll;
  std::vector<int>          *t_ietaGood;

  // List of branches
  TBranch                   *b_t_Tracks;        //!
  TBranch                   *b_t_TracksProp;    //!
  TBranch                   *b_t_TracksSaved;   //!
  TBranch                   *b_t_ietaAll;       //!
  TBranch                   *b_t_ietaGood;      //!

  GetEntries(std::string fname, std::string dirname, bool ifOld=true);
  virtual ~GetEntries();
  virtual Int_t    Cut(Long64_t entry);
  virtual Int_t    GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void     Init(TTree *tree);
  virtual void     Loop();
  virtual Bool_t   Notify();
  virtual void     Show(Long64_t entry = -1);

private:
  bool             ifOld_;
  TH1I            *h_tk[3], *h_eta[2];
  TH1D            *h_eff;
};

GetEntries::GetEntries(std::string fname, std::string dirnm, bool ifOld) : ifOld_(ifOld) {

  TFile      *file = new TFile(fname.c_str());
  TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
  std::cout << fname << " file " << file << " " << dirnm << " " << dir << std::endl;
  TTree      *tree = (TTree*)dir->Get("EventInfo");
  std::cout << "CalibTree " << tree << std::endl;
  Init(tree);
}

GetEntries::~GetEntries() {
  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

Int_t GetEntries::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

Long64_t GetEntries::LoadTree(Long64_t entry) {
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

void GetEntries::Init(TTree *tree) {
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
  fChain->SetBranchAddress("t_Tracks",      &t_Tracks,      &b_t_Tracks);
  fChain->SetBranchAddress("t_TracksProp",  &t_TracksProp,  &b_t_TracksProp);
  fChain->SetBranchAddress("t_TracksSaved", &t_TracksSaved, &b_t_TracksSaved);
  if (!ifOld_) {
    fChain->SetBranchAddress("t_ietaAll",     &t_ietaAll,     &b_t_ietaAll);
    fChain->SetBranchAddress("t_ietaGood",    &t_ietaGood,    &b_t_ietaGood);
  }
  Notify();

  h_tk[0] = new TH1I("Track0", "# of tracks produced",      2000, 0, 2000);
  h_tk[1] = new TH1I("Track1", "# of tracks propagated",    2000, 0, 2000);
  h_tk[2] = new TH1I("Track2", "# of tracks saved in tree", 2000, 0, 2000);
  h_eta[0] = new TH1I("Eta0", "i#eta (All Tracks)",           60, -30, 30);
  h_eta[1] = new TH1I("Eta1", "i#eta (Good Tracks)",          60, -30, 30);
  h_eff    = new TH1D("Eta2", "i#eta (Selection Efficiency)", 60, -30, 30);
}

Bool_t GetEntries::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}

void GetEntries::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain) return;
  fChain->Show(entry);
}

Int_t GetEntries::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void GetEntries::Loop() {
  //   In a ROOT session, you can do:
  //      Root > .L IsoTrkOfflineAnalyzer.C+g
  //      Root > GetEntries t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //

  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch
  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();
  
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    h_tk[0]->Fill(t_Tracks);
    h_tk[1]->Fill(t_TracksProp);
    h_tk[2]->Fill(t_TracksSaved);
    if (!ifOld_) {
      for (unsigned int k=0; k<t_ietaAll->size(); ++k)
	h_eta[0]->Fill((*t_ietaAll)[k]);
      for (unsigned int k=0; k<t_ietaGood->size(); ++k)
	h_eta[1]->Fill((*t_ietaGood)[k]);
    }
  }
  double ymaxk(0);
  if (!ifOld_) {
    for (int i=1; i<=h_eff->GetNbinsX(); ++i) {
      double rat(0), drat(0);
      if (h_eta[0]->GetBinContent(i) > ymaxk) ymaxk = h_eta[0]->GetBinContent(i);
      if (h_eta[1]->GetBinContent(i) > 0) {
	rat = h_eta[1]->GetBinContent(i)/h_eta[0]->GetBinContent(i);
	drat= rat*std::sqrt(pow((h_eta[1]->GetBinError(i)/h_eta[1]->GetBinContent(i)),2) +
			    pow((h_eta[0]->GetBinError(i)/h_eta[0]->GetBinContent(i)),2));
      }
      h_eff->SetBinContent(i,rat);
      h_eff->SetBinError(i,drat);
    }
  }
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(1110);       gStyle->SetOptTitle(0);
  int color[3] = {kBlack, kRed, kBlue};
  int lines[3] = {1, 2, 3};
  TCanvas *pad1 = new TCanvas("c_track", "c_track", 500, 500);
  pad1->SetRightMargin(0.10);
  pad1->SetTopMargin(0.10);
  pad1->SetFillColor(kWhite);
  std::string titl1[3] = {"Reconstructed", "Propagated", "Saved"};
  TLegend  *legend1 = new TLegend(0.55, 0.54, 0.90, 0.63);
  legend1->SetFillColor(kWhite);
  double ymax(0), xmax(0);
  for (int k=0; k<3; ++k) {
    int total(0), totaltk(0);
    for (int i=1; i<=h_tk[k]->GetNbinsX(); ++i) {
      if (ymax < h_tk[k]->GetBinContent(i)) ymax = h_tk[k]->GetBinContent(i);
      if (i > 1) total += (int)(h_tk[k]->GetBinContent(i));
      totaltk += (int)(h_tk[k]->GetBinContent(i))*(i-1);
      if (h_tk[k]->GetBinContent(i) > 0) {
	if (xmax < h_tk[k]->GetBinLowEdge(i)+h_tk[k]->GetBinWidth(i))
	  xmax = h_tk[k]->GetBinLowEdge(i)+h_tk[k]->GetBinWidth(i);
      }
    }
    h_tk[k]->SetLineColor(color[k]);
    h_tk[k]->SetMarkerColor(color[k]);
    h_tk[k]->SetLineStyle(lines[k]);
    std::cout << h_tk[k]->GetTitle() << " Entries " << h_tk[k]->GetEntries()
	      << " Events " << total << " Tracks " << totaltk << std::endl;
    legend1->AddEntry(h_tk[k],titl1[k].c_str(),"l");
  }
  int i1 = (int)(0.1*xmax) + 1;
  xmax   = 10.0*i1;
  int i2 = (int)(0.01*ymax) + 1;
  
  ymax   = 100.0*i2;
  for (int k=0; k<3; ++k) {
    h_tk[k]->GetXaxis()->SetRangeUser(0,  xmax);
    h_tk[k]->GetYaxis()->SetRangeUser(0.1,ymax);
    h_tk[k]->GetXaxis()->SetTitle("# Tracks");
    h_tk[k]->GetYaxis()->SetTitle("Events");
    h_tk[k]->GetYaxis()->SetLabelOffset(0.005);
    h_tk[k]->GetYaxis()->SetTitleOffset(1.20);
    if (k == 0) h_tk[k]->Draw("hist");
    else        h_tk[k]->Draw("hist sames");
  }
  pad1->Update();
  pad1->SetLogy();
  ymax = 0.90;
  for (int k=0; k<3; ++k) {
    TPaveStats* st1 = (TPaveStats*)h_tk[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != NULL) {
      st1->SetLineColor(color[k]);
      st1->SetTextColor(color[k]);
      st1->SetY1NDC(ymax-0.09); st1->SetY2NDC(ymax);
      st1->SetX1NDC(0.55);      st1->SetX2NDC(0.90);
      ymax -= 0.09;
    }
  }
  pad1->Modified();
  pad1->Update();
  legend1->Draw("same");
  pad1->Update();

  if (!ifOld_) {
    TCanvas *pad2 = new TCanvas("c_ieta", "c_ieta", 500, 500);
    pad2->SetRightMargin(0.10);
    pad2->SetTopMargin(0.10);
    pad2->SetFillColor(kWhite);
    pad2->SetLogy();
    std::string titl2[2] = {"All Tracks", "Selected Tracks"};
    TLegend  *legend2 = new TLegend(0.55, 0.28, 0.90, 0.35);
    legend2->SetFillColor(kWhite);
    i2    = (int)(0.001*ymaxk) + 1;
    ymax  = 1000.0*i2;
    for (int k=0; k<2; ++k) {
      h_eta[k]->GetYaxis()->SetRangeUser(1,ymax);
      h_eta[k]->SetLineColor(color[k]);
      h_eta[k]->SetMarkerColor(color[k]);
      h_eta[k]->SetLineStyle(lines[k]);
      h_eta[k]->GetXaxis()->SetTitle("i#eta");
      h_eta[k]->GetYaxis()->SetTitle("Tracks");
      h_eta[k]->GetYaxis()->SetLabelOffset(0.005);
      h_eta[k]->GetYaxis()->SetTitleOffset(1.20);
      legend2->AddEntry(h_eta[k],titl2[k].c_str(),"l");
      if (k == 0) h_eta[k]->Draw("hist");
      else        h_eta[k]->Draw("hist sames");
    }
    pad2->Update();
    double ymin = 0.10;
    for (int k=0; k<2; ++k) {
      TPaveStats* st1 = (TPaveStats*)h_eta[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	st1->SetLineColor(color[k]);
	st1->SetTextColor(color[k]);
	st1->SetY1NDC(ymin); st1->SetY2NDC(ymin+0.09);
	st1->SetX1NDC(0.55); st1->SetX2NDC(0.90);
	ymin += 0.09;
      }
    }
    pad2->Modified();
    pad2->Update();
    legend2->Draw("same");
    pad2->Update();

    TCanvas *pad3 = new TCanvas("c_effi", "c_effi", 500, 500);
    pad3->SetRightMargin(0.10);
    pad3->SetTopMargin(0.10);
    pad3->SetFillColor(kWhite);
    pad3->SetLogy();
    h_eff->SetStats(0);
    h_eff->SetMarkerStyle(20);
    h_eff->GetXaxis()->SetTitle("i#eta");
    h_eff->GetYaxis()->SetTitle("Efficiency");
    h_eff->GetYaxis()->SetLabelOffset(0.005);
    h_eff->GetYaxis()->SetTitleOffset(1.20);
    h_eff->Draw();
    pad3->Modified();
    pad3->Update();
  }
}

template<class Hist> void 
DrawHist(std::vector<Hist*> hist, std::vector<int> comb, 
	 int m, bool typex, bool save) {
  std::string titles[27] = {"Data with Method 0", "Data with Method 2", 
			    "Data with Method 3", "Data with Method 0",
			    "Data with Method 2", "Data with Method 3", 
			    "Data with Method 0", "Data with Method 2", 
			    "Data with Method 3", "Data with Method 0",
			    "Data with Method 2", "Data with Method 3", 
			    "MC (#pi) Method 0",  "MC (#pi) Method 2",
                            "MC (#pi) Method 3",  "MC (#pi) No PU",
			    "MC (QCD) Method 0",  "MC (QCD) Method 0",
			    "MC (QCD) Method 0",  "MC (QCD) Method 0",
			    "p = 40:60 GeV Tracks (Method 0)", 
			    "p = 40:60 GeV Tracks (Method 2)", 
			    "p = 40:60 GeV Tracks (Method 3)", 
			    "p = 40:60 GeV Tracks (Method 0)", 
			    "p = 20:30 GeV Tracks (Method 0)",
			    "p = 40:60 GeV Tracks", "p = 20:30 GeV Tracks"};
  int colors[6]={1,2,4,7,6,9};
  int mtype[6]={20,21,22,23,24,33};
  std::string dtitl[17] = {"Data (25 ns) Method 0",  "Data (25 ns) Method 2",
			   "Data (25 ns) Method 3",  "Data (50 ns) Method 0",
			   "Data (50 ns) Method 2",  "Data (50 ns) Method 3",
			   "Data (B+C) Method 0",    "Data (B+C) Method 2",
			   "Data (B+C) Method 3",    "#pi MC (No PU) Method 0",
			   "#pi MC (No PU) Method 2","#pi MC (No PU) Method 3",
			   "#pi MC (25 ns) Method 0","#pi MC (25 ns) Method 2",
			   "#pi MC (25 ns) Method 3","QCD MC (25 ns) Method 0",
			   "QCD MC (50 ns) Method 0"};
  std::string stitl[30] = {"p 2:4 GeV all PV","p 4:7 GeV all PV",
			   "p 7:10 GeV all PV","p 10:15 GeV all PV",
			   "p 15:20 GeV all PV","p 20:30 GeV all PV",
			   "p 30:40 GeV all PV","p 40:60 GeV all PV",
			   "p 60:100 GeV all PV","all p all PV",
			   "p 2:4 GeV PV 0:12","p 4:7 GeV PV 0:12",
			   "p 7:10 GeV PV 0:12","p 10:15 GeV PV 0:12",
			   "p 15:20 GeV PV 0:12","p 20:30 GeV PV 0:12",
			   "p 30:40 GeV PV 0:12","p 40:60 GeV PV 0:12",
			   "p 60:100 GeV PV 0:12","all p PV 0:12",
			   "p 2:4 GeV PV > 12","p 4:7 GeV PV > 12",
			   "p 7:10 GeV PV > 12","p 10:15 GeV PV > 12",
			   "p 15:20 GeV PV > 12","p 20:30 GeV PV > 12",
			   "p 30:40 GeV PV > 12","p 40:60 GeV PV > 12",
			   "p 60:100 GeV PV > 12","all p PV > 12"};

  char name[20];
  std::vector<double> resultv, resulte;
  for (unsigned int j=0; j<hist.size(); ++j) {
    hist[j]->SetTitle(titles[m].c_str());
    hist[j]->SetLineColor(colors[j]);
    hist[j]->SetMarkerColor(colors[j]);
    hist[j]->SetMarkerStyle(mtype[j]);
    hist[j]->GetYaxis()->SetRangeUser(0.4,1.6);
    hist[j]->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
    hist[j]->GetXaxis()->SetTitle("i#eta");
    /*
    if (!typex) {
      Hist* hist1 = (Hist*)hist[j]->Clone();
      TFitResultPtr Fit = hist1->Fit("pol0","+WWQRD","");
      resultv.push_back(Fit->Value(1));
      std::cout << "Fit " << Fit->Value(1) << std::endl;
    }
    */
  }
  if (typex) sprintf (name, "c_respX%d", m);
  else       sprintf (name, "c_respZ%d", m);
  TCanvas *pad = new TCanvas(name, name, 700, 500);
  pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
  double yl = 0.75 - 0.025*hist.size();
  TLegend *legend = new TLegend(0.40, yl, 0.90, 0.75);
  legend->SetFillColor(kWhite);
  for (unsigned int j=0; j<hist.size(); ++j) {
    if (j == 0) hist[j]->Draw("");
    else        hist[j]->Draw("sames");
    char title[100];
    int j1 = comb[j]/100 - 1;
    int j2 = comb[j]%100 - 1;
    sprintf (title, "%s (%s)", dtitl[j1].c_str(), stitl[j2].c_str());
    legend->AddEntry(hist[j],title,"lp");
  }
  pad->Update();
  double xmax = 0.90;
  for (unsigned int k=0; k<hist.size(); ++k) {
    TPaveStats* st1 = (TPaveStats*)hist[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != NULL) {
      st1->SetLineColor(colors[k]);
      st1->SetTextColor(colors[k]);
      st1->SetY1NDC(0.78); st1->SetY2NDC(0.90);
      st1->SetX1NDC(xmax-0.12); st1->SetX2NDC(xmax);
      xmax -= 0.12;
    }
  }
  if (!typex) {
    xmax = 0.90;
    double ymax = 0.895;
    int nbox(0);
    for (unsigned int k=0; k<hist.size(); ++k) {
      double mean(0), rms(0), ent(0);
      for (int i=1; i<=hist[k]->GetNbinsX(); ++i) {
	double error = hist[k]->GetBinError(i);
	double value = hist[k]->GetBinContent(i);
	mean += (value/(error*error));
	rms  += (value*value/(error*error));
	ent  += (1.0/(error*error));
      }
      mean /= ent;
      rms  /= ent;
      double err = std::sqrt((rms-mean*mean)/std::sqrt(ent));
      TPaveText *txt1 = new TPaveText(xmax-0.20,ymax-0.04,xmax,ymax,"blNDC");
      txt1->SetFillColor(0);
      txt1->SetLineColor(1);
      txt1->SetTextColor(colors[k]);
      char txt[80];
      sprintf (txt, "Mean = %5.3f #pm %5.3f", mean, err);
      txt1->AddText(txt);
      txt1->Draw("same");
      xmax -= 0.20; nbox++;
      if (nbox == 3) {
	ymax -= 0.04; xmax = 0.90; nbox = 0;
      }
      pad->Modified();
      pad->Update();
    }
  }
  pad->Modified();
  pad->Update();
  legend->Draw("same");
  pad->Update();
  if (save) {
    sprintf (name, "%s.gif", pad->GetName());
    pad->Print(name);
  }
}

void PlotHists(std::string fname, int mode, bool save=false, bool typex=true) {

  std::string dname[17] = {"D250","D252","D253","D500","D502","D503",
			   "DAL0","DAL2","DAL3","PNP0","PNP2","PNP3",
			   "P250","P252","P253","Q250","Q500"};
  std::string snamex[30] = {"X00","X01","X02","X03","X04","X05","X06","X07",
			    "X08","X09","X10","X11","X12","X13","X14","X15",
			    "X16","X17","X18","X19","X20","X21","X22","X23",
			    "X24","X25","X26","X27","X28","X29"};
  std::string snamez[30] = {"Z00","Z01","Z02","Z03","Z04","Z05","Z06","Z07",
			    "Z08","Z09","Z10","Z11","Z12","Z13","Z14","Z15",
			    "Z16","Z17","Z18","Z19","Z20","Z21","Z22","Z23",
			    "Z24","Z25","Z26","Z27","Z28","Z29"};

  int comb[210] = {108,118,128,408,418,428,
		   208,218,228,508,518,528,
		   308,318,328,608,618,628,
		   706,702,708,704,0,0,
		   806,802,808,804,0,0,
		   906,902,908,904,0,0,
		   708,718,728,0,0,0,
		   808,818,828,0,0,0,
		   908,918,928,0,0,0,
		   706,716,726,0,0,0,
		   806,816,826,0,0,0,
		   906,916,926,0,0,0,
		   1308,2718,2728,0,0,0,
		   1408,1418,1428,0,0,0,
		   1508,1518,1528,0,0,0,
		   1008,1108,1208,0,0,0,
		   1608,1618,1628,0,0,0,
		   1708,1718,1728,0,0,0,
		   1606,1616,1626,0,0,0,
		   1706,1716,1726,0,0,0,
		   708,718,728,2708,2718,2728,
		   808,818,828,1408,1418,1428,
		   908,918,928,1508,1518,1528,
		   708,1608,1708,1008,0,0,
		   706,1606,1706,0,0,0,
                   1308,1408,1508,0,0,0,
		   1306,1406,1506,0,0,0,
		   701,711,721,0,0,0,
		   702,712,722,0,0,0,
		   703,713,723,0,0,0,
		   704,714,724,0,0,0,
		   705,715,725,0,0,0,
		   707,717,727,0,0,0,
		   709,719,729,0,0,0,
		   710,720,730,0,0,0};
  int ncombs[35]={6,6,6,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,6,6,6,4,3,3,3,
		  3,3,3,3,3,3,3,3};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  if (typex) {
    gStyle->SetOptStat(1100);
  } else {
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(11);
  }
  TFile      *file = new TFile(fname.c_str());
  int modemin = (mode < 0 || mode > 34) ? 0 : mode;
  int modemax = (mode < 0 || mode > 34) ? 34 : mode;
  for (int m=modemin; m <= modemax; ++m) {
    int nh = ncombs[m];
    std::vector<TProfile*> histp;
    std::vector<TH1D*>     histh;
    std::vector<int>       icomb;
    char name[20];
    bool ok(true);
    for (int j=0; j<nh; ++j) {
      int j1 = comb[m*6+j]/100 - 1;
      int j2 = comb[m*6+j]%100 - 1;
      icomb.push_back(comb[m*6+j]);
      if (typex) {
	sprintf (name,"%seta%s",dname[j1].c_str(),snamex[j2].c_str());
	TProfile* hist1 = (TProfile*)file->FindObjectAny(name);
	std::cout << name << " read out at " << hist1 << std::endl;
	if (hist1 != 0) {
	  TProfile* hist  = (TProfile*)hist1->Clone();
	  histp.push_back(hist);
	} else {
	  ok = false;
	}
      } else {
	sprintf (name,"%s%s",dname[j1].c_str(),snamez[j2].c_str());
	TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
	std::cout << name << " read out at " << hist1 << std::endl;
	if (hist1 != 0) {
	  TH1D* hist  = (TH1D*)hist1->Clone();
	  histh.push_back(hist);
	} else {
	  ok = false;
	}
      }
    }
    std::cout << "Mode " << m << " Flag " << ok << std::endl;
    if (ok) {
      if (typex) DrawHist(histp, icomb, m, true, save);
      else       DrawHist(histh, icomb, m, false, save);
    }
  }
}

std::pair<double,double> GetMean(TH1D* hist, double xmin, double xmax) {

  double mean(0), rms(0), err(0), wt(0);
  for (int i=1; i<=hist->GetNbinsX(); ++i) {
    if (((hist->GetBinLowEdge(i)) >= xmin) || 
	((hist->GetBinLowEdge(i)+hist->GetBinWidth(i)) <= xmax)) {
      double cont = hist->GetBinContent(i);
      double valu = hist->GetBinLowEdge(i)+0.5*+hist->GetBinWidth(i);
      wt         += cont;
      mean       += (valu*cont);
      rms        += (valu*valu*cont);
    }
  }
  if (wt > 0) {
    mean /= wt;
    rms  /= wt;
    err   = std::sqrt((rms-mean*mean)/wt);
  }
  return std::pair<double,double>(mean,err);
}

void FitHists(std::string infile, std::string outfile, std::string dname,
	      int mode, bool append=true, bool saveAll=false) {

  int iname[10]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int checkmode[10] = {1000, 1000, 1000, 1000, 1000, 10, 1000, 100, 1000, 1};
  double xbins[9]   = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
  double vbins[6]   = {0.0, 7.0, 10.0, 13.0, 16.0, 50.0};
  double dlbins[9]  = {0.0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};
  std::string sname[4] = {"ratio","etaR", "dl1R","nvxR"};
  std::string lname[4] = {"Z", "E", "L", "V"};
  int         numb[4]  = {8, 8, 8, 5};
  TFile      *file = new TFile(infile.c_str());
  std::vector<TH1D*> hists;
  char name[100];
  if (file != 0) {
    for (int m1=0; m1<4; ++m1) {
      for (int m2=0; m2<10; ++m2) {
	sprintf (name, "%s%s%d0", dname.c_str(), sname[m1].c_str(), iname[m2]);
	TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
	bool ok = ((hist0 != 0) && (hist0->GetEntries() > 25));
	if ((mode/checkmode[m2])%10 > 0 && ok) {
	  TH1D* histo(0);
	  for (int j=0; j<=numb[m1]; ++j) {
	    sprintf (name, "%s%s%d%d", dname.c_str(), sname[m1].c_str(), iname[m2], j);
	    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
	    TH1D* hist  = (TH1D*)hist1->Clone();
	    double value(0), error(0);
	    if (hist->GetEntries() > 0) {
	      value = hist->GetMean(); error = hist->GetRMS();
	    }
	    if (j == 0) {
	      sprintf (name, "%s%s%d", dname.c_str(), lname[m1].c_str(), iname[m2]);
	      if (m1 <= 1)      histo = new TH1D(name, hist->GetTitle(), numb[m1], xbins);
	      else if (m1 == 2) histo = new TH1D(name, hist->GetTitle(), numb[m1], dlbins);
	      else              histo = new TH1D(name, hist->GetTitle(), numb[m1], vbins);
	    }
	    if (hist->GetEntries() > 4) {
	      double mean = hist->GetMean(), rms = hist->GetRMS();
	      double LowEdge = mean - 1.5*rms;
	      double HighEdge = mean + 2.0*rms;
	      if (LowEdge < 0.15) LowEdge = 0.15;
	      char option[20];
	      if (hist0->GetEntries() > 100) sprintf (option, "+QRS");
	      else                           sprintf (option, "+QRWLS");
	      double minvalue(0.30);
	      if (iname[m2] == 0) {
		sprintf (option, "+QRWLS");
		HighEdge = 0.9;
		minvalue = 0.10;
	      }
	      TFitResultPtr Fit = hist->Fit("gaus",option,"",LowEdge,HighEdge);
	      value = Fit->Value(1);
	      error = Fit->FitResult::Error(1); 
	      std::pair<double,double> meaner = GetMean(hist,0.2,2.0);
//	      std::cout << "Fit " << value << ":" << error << ":" << hist->GetMeanError() << " Mean " << meaner.first << ":" << meaner.second;
	      if (value < minvalue || value > 2.0 || error > 0.5) {
		value = meaner.first; error = meaner.second;
	      }
//	      std::cout << " Final " << value << ":" << error << std::endl;
	    }
	    if (j == 0) {
	      hists.push_back(hist);
	    } else {
	      if (saveAll) hists.push_back(hist);
	      histo->SetBinContent(j, value);
	      histo->SetBinError(j, error);
	      if (j == numb[m1]) {
		hists.push_back(histo);
	      }
	    }
	  }
	}
      }
    }
    TFile* theFile(0);
    if (append) {
      theFile = new TFile(outfile.c_str(), "UPDATE");
    } else {
      theFile = new TFile(outfile.c_str(), "RECREATE");
    }

    theFile->cd();
    for (unsigned int i=0; i<hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void FitCombineHist(std::string infile, std::string outfile, std::string dname1,
		    std::string dname2, std::string dname, bool append=true) {

  double xbins[43]= {-21.5,-20.5,-19.5,-18.5,-17.5,-16.5,-15.5,-14.5,-13.5,
		     -12.5,-11.5,-10.5,-9.5,-8.5,-7.5,-6.5,-5.5,-4.5,-3.5,
		     -2.5,-1.5,0.0,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,
		     11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,21.5};
  std::string sname("ratio"), lname("Z");
  int         iname(7), numb(42);

  TFile      *file = new TFile(infile.c_str());
  std::vector<TH1D*> hists;
  char name[100];
  if (file != 0) {
    sprintf (name, "%s%s%d0", dname1.c_str(), sname.c_str(), iname);
    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
    bool ok1 = (hist1 != 0);
    sprintf (name, "%s%s%d0", dname2.c_str(), sname.c_str(), iname);
    TH1D* hist2 = (TH1D*)file->FindObjectAny(name);
    bool ok2 = (hist2 != 0);
    if (ok1 || ok2) {
      int    nbin;
      double xmin, xmax;
      if (ok1) {
	nbin = hist1->GetNbinsX();
	xmin = hist1->GetBinLowEdge(1);
	xmax = hist1->GetBinLowEdge(nbin)+hist1->GetBinWidth(nbin);
      } else {
	nbin = hist2->GetNbinsX();
	xmin = hist2->GetBinLowEdge(1);
	xmax = hist2->GetBinLowEdge(nbin)+hist1->GetBinWidth(nbin);
      }
      TH1D* hist0(0);
      TH1D* histo(0);
      for (int j=0; j<=numb; ++j) {
	sprintf (name, "%s%s%d%d", dname1.c_str(), sname.c_str(), iname, j);
	hist1 = (TH1D*)file->FindObjectAny(name);
	sprintf (name, "%s%s%d%d", dname2.c_str(), sname.c_str(), iname, j);
	hist2 = (TH1D*)file->FindObjectAny(name);
	if (hist1 != 0) {
	  nbin = hist1->GetNbinsX();
	  xmin = hist1->GetBinLowEdge(1);
	  xmax = hist1->GetBinLowEdge(nbin)+hist1->GetBinWidth(nbin);
	} else {
	  nbin = hist2->GetNbinsX();
	  xmin = hist2->GetBinLowEdge(1);
	  xmax = hist2->GetBinLowEdge(nbin)+hist1->GetBinWidth(nbin);
	}
	TH1D* hist;
	sprintf (name, "%s%s%d%d", dname.c_str(), sname.c_str(), iname, j);
	if (hist1 != 0) hist = new TH1D(name, hist1->GetTitle(),nbin,xmin,xmax);
	else            hist = new TH1D(name, hist2->GetTitle(),nbin,xmin,xmax);
	double total(0);
	for (int k=1; k<=nbin; ++k) {
	  double value(0), error(0);
	  if (ok1) {
	    value += (hist1->GetBinContent(k)); 
	    error += ((hist1->GetBinError(k))*(hist1->GetBinError(k)));
	  }
	  if (ok2) {
	    value += (hist2->GetBinContent(k)); 
	    error += ((hist2->GetBinError(k))*(hist2->GetBinError(k)));
	  }
	  hist->SetBinContent(k,value); hist->SetBinError(k,sqrt(error));
	  total += value;
	}
	double value(0), error(0);
	if (hist->GetEntries() > 0) {
	  value = hist->GetMean(); error = hist->GetRMS();
	}
	if (j == 0) {
	  hist0 = hist;
	  sprintf (name, "%s%s%d", dname.c_str(), lname.c_str(), iname);
	  histo = new TH1D(name, hist->GetTitle(), numb, xbins);
	}
	if (total > 4) {
	  double mean = hist->GetMean(), rms = hist->GetRMS();
	  double LowEdge = mean - 1.5*rms;
	  double HighEdge = mean + 2.0*rms;
	  if (LowEdge < 0.15) LowEdge = 0.15;
	  char option[20];
	  if (total > 100) {
	    sprintf (option, "+QRS");
	  } else {
            sprintf (option, "+QRWLS");
	    HighEdge= mean+1.5*rms;
	  }
	  double minvalue(0.30);
	  TFitResultPtr Fit = hist->Fit("gaus",option,"",LowEdge,HighEdge);
	  value = Fit->Value(1);
	  error = Fit->FitResult::Error(1); 
	  std::pair<double,double> meaner = GetMean(hist,0.2,2.0);
//	  std::cout << "Fit " << value << ":" << error << ":" << hist->GetMeanError() << " Mean " << meaner.first << ":" << meaner.second;
	  if (value < minvalue || value > 2.0 || error > 0.5) {
	    value = meaner.first; error = meaner.second;
	  }
//	  std::cout << " Final " << value << ":" << error << std::endl;
	}
	hists.push_back(hist);
	if (j != 0) {
	  histo->SetBinContent(j, value);
	  histo->SetBinError(j, error);
	  if (j == numb) hists.push_back(histo);
	}
      }
    }
    TFile* theFile(0);
    if (append) {
      theFile = new TFile(outfile.c_str(), "UPDATE");
    } else {
      theFile = new TFile(outfile.c_str(), "RECREATE");
    }

    theFile->cd();
    for (unsigned int i=0; i<hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void PlotFits(std::string fname, int mode, int mode2=0, bool save=false) {

  std::string dname[48] = {"DJT0", "DJT2", "DEG0", "DEG2", "DV00", "DV02",
			   "D250", "D252", "D500", "D502", "DAL0", "DAL2",
			   "DAL3", "D120", "DV10", "DV12", "DV20", "DV22",
			   "DV30", "DV32", "DNC0", "DNC2", "DHR0", "DHR2",
			   "DPR0", "DPR2", "D750", "D752", "D5B0", "D5B2",
			   "D5C0", "D5C2", "D5D0", "D5D2", "DED0", "DED2",
			   "DMD0", "DMD2", "Q250", "Q500", "PNP0", "PNP2",
			   "PNP3", "P250", "P252", "P253", "P750", "P752"};
  std::string snamex[10] = {"ratio00","ratio10","ratio20","ratio30",
			    "ratio40","ratio50","ratio60","ratio70",
			    "ratio80","ratio90"};
  std::string sname2[10] = {"Z0", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7",
			    "Z8", "Z9"};
  std::string dtitl[48] = {"Jet data (2015B,C,D) Method 0",
			   "Jet data (2015B,C,D) Method 2",
			   "e#gamma data (2015B,C,D) Method 0",
			   "e#gamma data (2015B,C,D) Method 2",
			   "(Jet + e#gamma) (2015B,C,D) Method 0",
			   "(Jet + e#gamma) (2015B,C,D) Method 2",
			   "Jet data (25 ns) Method 0", 
			   "Jet data (25 ns) Method 2",
			   "Jet data (50 ns) Method 0",
			   "Jet data (50 ns) Method 2",
			   "Jet data (B+C) Method 0",
			   "Jet data (B+C) Method 2",
			   "Jet data (B+C) Method 3",
			   "2012 Jet data",
			   "Jet data (2015B+C) Version 1 Method 0",
			   "Jet data (2015B+C) Version 1 Method 2",
			   "Jet data (2015B+C) Version 2 Method 0",
			   "Jet data (2015B+C) Version 2 Method 2",
			   "Jet data (2015B+C) Version 3 Method 0",
			   "Jet data (2015B+C) Version 3 Method 2",
			   "Jet data (New Constants) Method 0",
			   "Jet data (New Constants) Method 2",
			   "Jet data (Reference) Method 0",
			   "Jet data (Reference) Method 2",
			   "Jet data (Old Reference) Method 0",
			   "Jet data (Old Reference) Method 2",
			   "Jet data (75X) Method 0",
			   "Jet data (75X) Method 2",
			   "Jet data (2015B) Method 0",
			   "Jet data (2015B) Method 2",
			   "Jet data (2015C) Method 0",
			   "Jet data (2015C) Method 2",
			   "Jet data (2015D) Method 0",
			   "Jet data (2015D) Method 2",
			   "e#gamma data (2015D) Method 0",
			   "e#gamma data (2015D) Method 2",
			   "Single #mu data Method 0",
			   "Single #mu data Method 2",
			   "QCD MC (25 ns) Method 0",
			   "QCD MC (50 ns) Method 0",
			   "#pi MC (No PU) Method 0",
			   "#pi MC (No PU) Method 2",
			   "#pi MC (No PU) Method 3",
			   "#pi MC (25 ns) Method 0",
			   "#pi MC (25 ns) Method 2",
			   "#pi MC (25 ns) Method 3",
			   "#pi MC (25 ns) Method 0",
			   "#pi MC (25 ns) Method 2"};
  std::string stitl[10] = {"p 2:4 GeV","p 4:7 GeV",
			   "p 7:10 GeV","p 10:15 GeV",
			   "p 15:20 GeV","p 20:30 GeV",
			   "p 30:40 GeV","p 40:60 GeV",
			   "p 60:100 GeVV","all p"};
  std::string titles[40] = {"p 20:30 GeV Method 0", "p 40:60 GeV Method 0", 
			    "All p Method 0",       "p 40:60 GeV Method 0",
			    "p 40:60 GeV Method 2", "p 20:30 GeV Method 0",
			    "p 20:30 GeV Method 2", "p 20:30 GeV Method 0",
			    "p 40:60 GeV Method 0", "All p Method 0",
			    "p 20:30 GeV Method 2", "p 40:60 GeV Method 2", 
			    "All p Method 2",       "p 20:30 GeV Method 2",
			    "p 40:60 GeV Method 2", "All p Method 2",
			    "Method 0 (p 40:60 GeV)","Method 2 (p 40:60 GeV)",
			    "Method 2 (p 40:60 GeV)",
			    "#pi MC (No PU) p 20:30 GeV",
			    "#pi MC (No PU) p 40:60 GeV",
			    "#pi MC (No PU) all p",
			    "#pi MC (25 ns) p 20:30 GeV",
			    "#pi MC (25 ns) p 40:60 GeV",
			    "#pi MC (25 ns) all p",
			    "Data (p 40:60 GeV)", "Data (p 40:60 GeV)",
			    "Data (p 40:60 GeV)", "Data (p 40:60 GeV)",
			    "MC (p 40:60 GeV)",   "Data (p 2:4 GeV)", 
			    "Data (p 4:7 GeV)",   "Data (p 7:10 GeV)", 
			    "Data (p 10:15 GeV)", "Data (p 15:20 GeV)",
			    "Data (p 30:40 GeV)", "Data (p 60:100 GeV)",
			    "Data (All momenta)", "Method 0 (p 40:60 GeV)"
			    "Method 2 (p 40:60 GeV)"};
  int comb[240] = {1106,706,906,4406,3906,4006,
		   1108,708,908,4408,3908,4008,
		   1110,710,910,4410,3910,4010,
		   1108,2708,2108,2308,2508,0,
		   1208,2808,2208,2408,2608,0,
		   1106,2706,2106,2306,2506,0,
		   1206,2806,2206,2406,2606,0,
		   2906,3106,3306,1406,0,0,
		   2908,3108,3308,1408,0,0,
		   2910,3110,3310,1410,0,0,
		   3006,3206,3406,0,0,0,
		   3008,3208,3408,0,0,0,
		   3010,3210,3410,0,0,0,
		   1206,806,1006,0,0,0,
		   1208,808,1008,0,0,0,
		   1210,810,1010,0,0,0,
		   508,4708,0,0,0,0,
		   608,4808,0,0,0,0,
		   608,4508,0,0,0,0,
		   4106,4206,4306,0,0,0,
		   4108,4208,4308,0,0,0,
		   4110,4210,4310,0,0,0,
		   4406,4506,4606,4806,0,0,
		   4408,4508,4608,4808,0,0,
		   4410,4510,4610,4810,0,0,
		   1108,1208,0,0,0,0,
		   1508,1708,1908,0,0,0,
		   1608,1808,2008,0,0,0,
		   1508,1608,1708,1808,1908,2008,
		   3908,4008,0,0,0,0,
		   1101,1201,0,0,0,0,
		   1102,1202,0,0,0,0,
		   1103,1203,0,0,0,0,
		   1104,1204,0,0,0,0,
		   1105,1205,0,0,0,0,
		   1107,1207,0,0,0,0,
		   1109,1209,0,0,0,0,
		   1110,1210,0,0,0,0,
		   108,308,4708,0,0,0,
		   208,408,4808,0,0,0};
  int ncombs[40]={6,6,6,5,5,5,5,3,3,3,3,3,3,3,3,3,2,2,2,3,3,3,4,4,4,
		  2,3,3,6,2,2,2,2,2,2,2,2,2,3,3};
  int colors[6]={1,2,4,7,6,9};
  int mtype[6]={20,21,22,23,24,33};
  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  if (mode2 == 0) gStyle->SetOptStat(1110);
  else            gStyle->SetOptStat(10);
  gStyle->SetOptFit(1);

  TFile      *file = new TFile(fname.c_str());
  int modemin = (mode < 0 || mode > 39) ? 0 : mode;
  int modemax = (mode < 0 || mode > 39) ? 39 : mode;
  for (int m=modemin; m <= modemax; ++m) {
    int nh = ncombs[m];
    TH1D* hist[6];
    char name[40], namen[50];
    bool ok(true);
    double total[6];
    double ylow (0), yhigh(0), totalf(0);
    int ngood(0);
    for (int j=0; j<nh; ++j) {
      int j1 = comb[m*6+j]/100 - 1;
      int j2 = comb[m*6+j]%100 - 1;
      if (mode2 == 0)sprintf (name,"%s%s",dname[j1].c_str(),snamex[j2].c_str());
      else           sprintf (name,"%s%s",dname[j1].c_str(),sname2[j2].c_str());
      TH1D* h = (TH1D*)file->FindObjectAny(name);
      total[j] = 0;
      if (h != 0) {
	ngood++;
	sprintf (namen, "%sX", name);
	int    nbin = h->GetNbinsX();
	double xlow = h->GetBinLowEdge(1);
	double xhigh=  h->GetBinLowEdge(nbin)+h->GetBinWidth(nbin);
	hist[j] = new TH1D(namen,h->GetTitle(),nbin,xlow,xhigh);
	for (int i=1; i<=h->GetNbinsX(); ++i) {
	  double value = h->GetBinContent(i);
	  hist[j]->SetBinContent(i,value);
	  hist[j]->SetBinError(i,h->GetBinError(i));
	  if (mode2 == 0) {
	    if (h->GetBinLowEdge(i) >= 0.25 &&
		h->GetBinLowEdge(i) < 2.0) total[j] += value;
	  } else {
	    total[j] += value;
	  }
	}
	if (ngood == 1) totalf = total[j];
	if (mode2 == 0) {
	  double scale = (total[j] > 0) ? totalf/total[j] : 1.0;
	  for (int i=1; i<=hist[j]->GetNbinsX(); ++i) {
	    hist[j]->SetBinContent(i,scale*hist[j]->GetBinContent(i));
	    hist[j]->SetBinError(i,scale*hist[j]->GetBinError(i));
	    if (hist[j]->GetBinLowEdge(i) >= 0.25 &&
		hist[j]->GetBinLowEdge(i) < 2.0) {
	      if ((hist[j]->GetBinContent(i)) > yhigh)
		yhigh = (hist[j]->GetBinContent(i));
	    }
	  }
	} else {
	  yhigh = 1.6;
	}
	if (mode2 == 0) {
	  hist[j]->GetXaxis()->SetRangeUser(0.25,2.0);
	  hist[j]->GetXaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
	  hist[j]->GetYaxis()->SetTitle("Tracks");
	  double mean = hist[j]->GetMean(), rms = hist[j]->GetRMS();
	  double LowEdge = mean - 1.5*rms;
	  double HighEdge = mean + 2.0*rms;
	  if (LowEdge < 0.15) LowEdge = 0.15;
	  char option[20];
	  if (total[j] > 100) sprintf (option, "+QRS");
	  else                sprintf (option, "+QRWLS");
	  if (j2 <= 1) {
	    sprintf (option, "+QRWLS");
	    HighEdge = 0.9;
	  }
	  hist[j]->Fit("gaus",option,"",LowEdge,HighEdge);
	  hist[j]->GetFunction("gaus")->SetLineColor(colors[j]);
	} else {
	  hist[j]->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
	  hist[j]->GetXaxis()->SetTitle("i#eta");
	  hist[j]->Fit("pol0","q");
	  hist[j]->GetFunction("pol0")->SetLineColor(colors[j]);
	}
	hist[j]->SetTitle(titles[m].c_str());
	hist[j]->SetLineColor(colors[j]);
	hist[j]->SetMarkerColor(colors[j]);
	hist[j]->SetMarkerStyle(mtype[j]);
      } else {
	ok = false;
      }
    }
    if (mode2 == 0) {
      if (yhigh < 150.) {
	int iy = int(0.1*yhigh)+2;
	yhigh  = double(iy*10);
      } else {
	int iy = int(0.01*yhigh)+2;
	yhigh  = double(iy*100);
      }
    } else {
      ylow = 0.4;
    }
    if (ngood > 0) {
      sprintf (name, "c_fitres%d", m);
      TCanvas *pad = new TCanvas(name, name, 700, 500);
      pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
      double ymin = (mode2==0) ? 0.76 : 0.84;
      double yminl= ymin - ngood*0.025 - 0.01;
      TLegend *legend = new TLegend(0.50, yminl, 0.90, ymin-0.01);
      legend->SetFillColor(kWhite);
      bool first(true);
      for (int j=0; j<nh; ++j) {
	if (hist[j] != 0) {
	  hist[j]->GetYaxis()->SetRangeUser(ylow,yhigh);
	  if (first) hist[j]->Draw("");
	  else       hist[j]->Draw("sames");
	  first = false;
	  char title[100];
	  int j1 = comb[m*6+j]/100 - 1;
	  int j2 = comb[m*6+j]%100 - 1;
	  sprintf (title, "%s (%s)", dtitl[j1].c_str(), stitl[j2].c_str());
	  legend->AddEntry(hist[j],title,"lp");
	}
      }
      pad->Update();
      double xmax = 0.90;
      for (int k=0; k<nh; ++k) {
	if (hist[k] != 0) {
	  TPaveStats* st1 = (TPaveStats*)hist[k]->GetListOfFunctions()->FindObject("stats");
	  if (st1 != NULL) {
	    st1->SetLineColor(colors[k]);
	    st1->SetTextColor(colors[k]);
	    st1->SetY1NDC(ymin);      st1->SetY2NDC(0.90);
	    st1->SetX1NDC(xmax-0.12); st1->SetX2NDC(xmax);
	    xmax -= 0.12;
	  }
	}
      }
      pad->Modified();
      pad->Update();
      legend->Draw("same");
      pad->Update();
      if (save) {
	sprintf (name, "%s.gif", pad->GetName());
	pad->Print(name);
      }
    }
  }
}

void PlotHist(std::string fname, int num, int mode, bool save) {

  std::string name1[48] = {"DJT0", "DJT2", "DEG0", "DEG2", "DV00", "DV02",
			   "D250", "D252", "D500", "D502", "DAL0", "DAL2",
			   "DAL3", "D120", "DV10", "DV12", "DV20", "DV22",
			   "DV30", "DV32", "DNC0", "DNC2", "DHR0", "DHR2",
			   "DPR0", "DPR2", "D750", "D752", "D5B0", "D5B2",
			   "D5C0", "D5C2", "D5D0", "D5D2", "DED0", "DED2",
			   "DMD0", "DMD2", "Q250", "Q500", "PNP0", "PNP2",
			   "PNP3", "P250", "P252", "P253", "P750", "P752"};
  std::string name2[40] = {"Z7", "E7", "L7", "V7", "Z9", "E9", "L9", "V9",
			   "Z5", "E5", "L5", "V5", "Z6", "E6", "L6", "V6",
			   "Z8", "E8", "L8", "V8", "Z4", "E4", "L4", "V4",
			   "Z3", "E3", "L3", "V3", "Z2", "E2", "L2", "V2",
			   "Z1", "E1", "L1", "V1", "Z0", "E0", "L0", "V0"};
  std::string name3[40] = {"ratio70", "etaR70", "dl1R70", "nvxR70",
			   "ratio90", "etaR90", "dl1R90", "nvxR90",
			   "ratio50", "etaR50", "dl1R50", "nvxR50",
			   "ratio60", "etaR60", "dl1R60", "nvxR60",
			   "ratio80", "etaR80", "dl1R80", "nvxR80",
			   "ratio40", "etaR40", "dl1R40", "nvxR40",
			   "ratio30", "etaR30", "dl1R30", "nvxR30",
			   "ratio20", "etaR20", "dl1R20", "nvxR20",
			   "ratio10", "etaR10", "dl1R10", "nvxR10",
			   "ratio00", "etaR00", "dl1R00", "nvxR00"};
  std::string name4[29] = {"ratio71", "ratio72", "ratio73", "ratio74",
			   "ratio75", "ratio76", "ratio77", "ratio78",
			   "etaR71",  "etaR72",  "etaR73",  "etaR74",
			   "etaR75",  "etaR76",  "etaR77",  "etaR78",
			   "dl1R71",  "dl1R72",  "dl1R73",  "dl1R74",
			   "dl1R75",  "dl1R76",  "dl1R77",  "dl1R78",
			   "nvxR71",  "nvxR72",  "nvxR73",  "nvxR74",
			   "nvxR75"};
  std::string name5[42] = {"ratio71",  "ratio72",  "ratio73",  "ratio74",
			   "ratio75",  "ratio76",  "ratio77",  "ratio78",
			   "ratio79",  "ratio710", "ratio711", "ratio712",
			   "ratio713", "ratio714", "ratio715", "ratio716",
			   "ratio717", "ratio718", "ratio719", "ratio720",
			   "ratio721", "ratio722", "ratio723", "ratio724",
			   "ratio725", "ratio726", "ratio727", "ratio728",
			   "ratio729", "ratio730", "ratio731", "ratio732",
			   "ratio733", "ratio734", "ratio735", "ratio736",
			   "ratio737", "ratio738", "ratio739", "ratio740",
			   "ratio741", "ratio742"};
  std::string titl1[48] = {"Jet data (2015B,C,D) Method 0",
			   "Jet data (2015B,C,D) Method 2",
			   "e#gamma data (2015B,C,D) Method 0",
			   "e#gamma data (2015B,C,D) Method 2",
			   "(Jet + e#gamma) (2015B,C,D) Method 0",
			   "(Jet + e#gamma) (2015B,C,D) Method 2",
			   "Jet data (25 ns) Method 0", 
			   "Jet data (25 ns) Method 2",
			   "Jet data (50 ns) Method 0",
			   "Jet data (50 ns) Method 2",
			   "Jet data (B+C) Method 0",
			   "Jet data (B+C) Method 2",
			   "Jet data (B+C) Method 3",
			   "2012 Jet data",
			   "Jet data (2015B+C) Version 1 Method 0",
			   "Jet data (2015B+C) Version 1 Method 2",
			   "Jet data (2015B+C) Version 2 Method 0",
			   "Jet data (2015B+C) Version 2 Method 2",
			   "Jet data (2015B+C) Version 3 Method 0",
			   "Jet data (2015B+C) Version 3 Method 2",
			   "Jet data (New Constants) Method 0",
			   "Jet data (New Constants) Method 2",
			   "Jet data (Reference) Method 0",
			   "Jet data (Reference) Method 2",
			   "Jet data (Old Reference) Method 0",
			   "Jet data (Old Reference) Method 2",
			   "Jet data (75X) Method 0",
			   "Jet data (75X) Method 2",
			   "Jet data (2015B) Method 0",
			   "Jet data (2015B) Method 2",
			   "Jet data (2015C) Method 0",
			   "Jet data (2015C) Method 2",
			   "Jet data (2015D) Method 0",
			   "Jet data (2015D) Method 2",
			   "e#gamma data (2015D) Method 0",
			   "e#gamma data (2015D) Method 2",
			   "Single #mu data Method 0",
			   "Single #mu data Method 2",
			   "QCD MC (25 ns) Method 0",
			   "QCD MC (50 ns) Method 0",
			   "#pi MC (No PU) Method 0",
			   "#pi MC (No PU) Method 2",
			   "#pi MC (No PU) Method 3",
			   "#pi MC (25 ns) Method 0",
			   "#pi MC (25 ns) Method 2",
			   "#pi MC (25 ns) Method 3",
			   "#pi MC (25 ns) Method 0",
			   "#pi MC (25 ns) Method 2"};
  std::string titl2[40] = {"i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex",
			   "i#eta", "i#eta", "d_{L1}", "# Vertex"};
  static const int nmax1(154), nmax2(58), nmax3(84);
  int ncomb1[nmax1]     = {501,  502, 503, 504, 601, 602, 603, 604, 101, 102,
			   103,  104, 201, 202, 203, 204, 301, 302, 303, 304,
			   401,  402, 403, 404,4401,4402,4501,4502,4701,4702,
			   4801,4802, 509, 510, 511, 512, 609, 610, 611, 612,
			   109,  110, 111, 112, 209, 210, 211, 212, 309, 310,
			   311,  312, 409, 410, 411, 412, 701, 702, 801, 802,
			   901,  902,1001,1002,1101,1102,1201,1202,1401,1402,
			   1501,1502,1601,1602,1701,1702,1801,1802,1901,1902,
			   2001,2002,2101,2102,2201,2202,2301,2302,2401,2402,
			   2501,2502,2601,2602,2701,2702,2801,2802,2901,2902,
			   3001,3002,3101,3102,3201,3202,3301,3302,3401,3402,
			   3501,3502,3601,3602,3701,3702,3801,3802,3901,3902,
			   4001,4002,4101,4102,4201,4202,4301,4302,4601,4602,
			   505,  506, 507, 508, 605, 606, 607, 608, 105, 106,
			   107,  108, 205, 206, 207, 208, 305, 306, 307, 308,
			   405,  406, 407, 408};
  int ncomb2[nmax2]     = {201,202,203,204,205,206,207,208,209,210,
			   211,212,213,214,215,216,217,218,219,220,
			   221,222,223,224,225,226,227,228,229,401,
                           402,403,404,405,406,407,408,409,410,411,
                           412,413,414,415,416,417,418,419,420,421,
                           422,423,424,425,426,427,428,429};
  int ncomb3[nmax3]     = {601,  602, 603, 604, 605, 606, 607, 608, 609, 610,
			   611,  612, 613, 614, 615, 616, 617, 618, 619, 620,
			   621,  622, 623, 624, 625, 626, 627, 628, 629, 630,
                           631,  632, 633, 634, 635, 636, 637, 638, 639, 640,
                           641,  642,
			   4501,4502,4503,4504,4505,4506,4507,4508,4509,4510,
			   4511,4512,4513,4514,4515,4516,4517,4518,4519,4520,
			   4521,4522,4523,4524,4525,4526,4527,4528,4529,4530,
                           4531,4532,4533,4534,4535,4536,4537,4538,4539,4540,
                           4541,4542};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(1);
  int iopt(1110), nmax(nmax1-1);
  if (mode == 2) {
    iopt = 1100; nmax = (nmax2-1);
  } else if (mode == 3) {
    iopt = 1100; nmax = (nmax3-1);  gStyle->SetOptTitle(0);
  } else if (mode == 0) {
    iopt = 10;
  }
  gStyle->SetOptStat(iopt);  gStyle->SetOptFit(1);
  TFile      *file = new TFile(fname.c_str());
  char name[100], namep[100], title[100];
  int kmin = (num >= 0 && num <= nmax) ? num : 0;
  int kmax = (num >= 0 && num <= nmax) ? num : nmax;
  for (int k=kmin; k<=kmax; ++k) {
    int i1(0), i2(0);
    if (mode == 3) {
      i1 = ((ncomb3[k]/100)%100-1); i2 = ((ncomb3[k]%100)-1);
    } else if (mode == 2) {
      i1 = ((ncomb2[k]/100)%100-1); i2 = ((ncomb2[k]%100)-1);
    } else {
      i1 = ((ncomb1[k]/100)%100-1); i2 = ((ncomb1[k]%100)-1);
    }
    if (mode == 3) {
      int eta = ((k%42) < 21) ? ((k%42)-21) : ((k%42)-20);
      sprintf (title, "%s (i#eta = %d)", titl1[i1].c_str(), eta);
    } else {
      sprintf (title, "%s", titl1[i1].c_str());
    }
    if (mode == 0) {
      sprintf (name,  "%s%s",  name1[i1].c_str(), name2[i2].c_str());
      sprintf (namep, "%s%s%d",name1[i1].c_str(), name2[i2].c_str(), mode);
    } else if (mode == 3) {
      sprintf (name,  "%s%s",  name1[i1].c_str(), name5[i2].c_str());
      sprintf (namep, "%s%s%d",name1[i1].c_str(), name5[i2].c_str(), mode);
    } else if (mode == 2) {
      sprintf (name,  "%s%s",  name1[i1].c_str(), name4[i2].c_str());
      sprintf (namep, "%s%s%d",name1[i1].c_str(), name4[i2].c_str(), mode);
    } else {
      sprintf (name,  "%s%s",  name1[i1].c_str(), name3[i2].c_str());
      sprintf (namep, "%s%s%d",name1[i1].c_str(), name3[i2].c_str(), mode);
    }
    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
    if (hist1 != 0) {
      TH1D* hist = (TH1D*)(hist1->Clone()); 
      TCanvas *pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      if (mode == 0) {
	hist->GetXaxis()->SetTitle(titl2[i2].c_str());
	hist->GetYaxis()->SetTitle("<E_{HCAL}/(p-E_{ECAL})>");
	hist->GetYaxis()->SetRangeUser(0.4,1.6);
	hist->Fit("pol0","q");
      } else {
	hist->GetYaxis()->SetTitle("Tracks");
	hist->GetXaxis()->SetTitle("E_{HCAL}/(p-E_{ECAL})");
	hist->GetXaxis()->SetRangeUser(0.0,3.0);
      }
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleOffset(1.20);
      hist->Draw();
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
	double ymin = (mode == 0) ? 0.78 : 0.66; 
	st1->SetY1NDC(ymin); st1->SetY2NDC(0.90);
	st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
      }
      TPaveText *txt1 = new TPaveText(0.50,0.60,0.90,0.65,"blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      sprintf (txt, "%s", title);
      txt1->AddText(txt);
      txt1->Draw("same");
      pad->Modified();
      pad->Update();
      if (save) {
	sprintf (name, "%s.pdf", pad->GetName());
	pad->Print(name);
      }	
    }
  }
}

void doPlot(std::string outfile="histodata.root") {
  IsoTrkOfflineAnalyzer p1("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015.root","HcalIsoTrkAnalyzerM0","DV10",100,true);
  p1.Loop();
  p1.SavePlot(outfile,false);
  IsoTrkOfflineAnalyzer p2("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015.root","HcalIsoTrkAnalyzerM2","DV12",100,true);
  p2.Loop();
  p2.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p3("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_All.root","HcalIsoTrkAnalyzerM0","DV20",100,true);
  p3.Loop();
  p3.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p4("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_All.root","HcalIsoTrkAnalyzerM2","DV22",100,true);
  p4.Loop();
  p4.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p5("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_JetHT.root","HcalIsoTrkAnalyzerM0","DV30",100,true);
  p5.Loop();
  p5.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p6("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_JetHT.root","HcalIsoTrkAnalyzerM2","DV32",100,true);
  p6.Loop();
  p6.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p7("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/HLTnewcondition.root","HcalIsoTrkAnalyzerM0","DNC0",100,true);
  p7.Loop();
  p7.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p8("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/HLTnewcondition.root","HcalIsoTrkAnalyzerM2","DNC2",100,true);
  p8.Loop();
  p8.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p9("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/HLTreference.root","HcalIsoTrkAnalyzerM0","DHR0",100,true);
  p9.Loop();
  p9.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p10("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/HLTreference.root","HcalIsoTrkAnalyzerM2","DHR2",100,true);
  p10.Loop();
  p10.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p11("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/PRrefernce.root","HcalIsoTrkAnalyzerM0","DPR0",100,true);
  p11.Loop();
  p11.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer p12("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/PRrefernce.root","HcalIsoTrkAnalyzerM2","DPR2",100,true);
  p12.Loop();
  p12.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m1("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_pi50M2NoPU.root", "method0", "PNP0",200,false);
  m1.Loop();
  m1.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m2("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_pi50M2NoPU.root", "method2", "PNP2",200,false);
  m2.Loop();
  m2.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m3("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_pi50M3NoPU.root", "method3", "PNP3",200,false);
  m3.Loop();
  m3.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m4("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_pi50M2Bx25ns.root", "method0", "P250",200,false);
  m4.Loop();
  m4.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m5("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_pi50M2Bx25ns.root", "method2", "P252",200,false);
  m5.Loop();
  m5.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m6("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_pi50M3Bx25ns.root", "method3", "P253",200,false);
  m6.Loop();
  m6.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m7("/afs/cern.ch/work/g/gwalia/public/output_pi50_25nsBX_method2_753_2711.root", "method0", "P750",200,false);
  m7.Loop();
  m7.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m8("/afs/cern.ch/work/g/gwalia/public/output_pi50_25nsBX_method2_753_2711.root", "method2", "P752",200,false);
  m8.Loop();
  m8.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer q1("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_QCD25nsBX.root","method2","Q250",200,false);
  q1.Loop();
  q1.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer q2("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_QCD50nsBX.root","method2","Q500",200,false);
  q2.Loop();
  q2.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c0("/afs/cern.ch/work/g/gwalia/public/JetHT_2015B_75X.root","HcalIsoTrkAnalyzerM0","D750",100,true);
  c0.Loop();
  c0.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c1("/afs/cern.ch/work/g/gwalia/public/JetHT_2015B_75X.root","HcalIsoTrkAnalyzerM2","D752",100,true);
  c1.Loop();
  c1.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c2("/afs/cern.ch/work/g/gwalia/public/JetHT_76X_2015B.root","HcalIsoTrkAnalyzerM0","D5B0",100,true);
  c2.Loop();
  c2.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c3("/afs/cern.ch/work/g/gwalia/public/JetHT_76X_2015B.root","HcalIsoTrkAnalyzerM2","D5B2",100,true);
  c3.Loop();
  c3.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c4("/afs/cern.ch/work/g/gwalia/public/JetHT_76X_2015C.root","HcalIsoTrkAnalyzerM0","D5C0",100,true);
  c4.Loop();
  c4.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c5("/afs/cern.ch/work/g/gwalia/public/JetHT_76X_2015C.root","HcalIsoTrkAnalyzerM2","D5C2",100,true);
  c5.Loop();
  c5.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c6("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/2015DJSON1.root","HcalIsoTrkAnalyzerM0","D5D0",100,true);
  c6.Loop();
  c6.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer c7("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/2015DJSON1.root","HcalIsoTrkAnalyzerM2","D5D2",100,true);
  c7.Loop();
  c7.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d1("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_25nsAll.root","HcalIsoTrkAnalyzerM0","D250",100,true);
  d1.Loop();
  d1.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d2("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_25nsAll.root","HcalIsoTrkAnalyzerM2","D252",100,true);
  d2.Loop();
  d2.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d3("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_50nsAll.root","HcalIsoTrkAnalyzerM0","D500",100,true);
  d3.Loop();
  d3.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d4("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_50nsAll.root","HcalIsoTrkAnalyzerM2","D502",100,true);
  d4.Loop();
  d4.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d5("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_All.root","HcalIsoTrkAnalyzerM0","DAL0",100,true);
  d5.Loop();
  d5.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d6("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/output_2015_All.root","HcalIsoTrkAnalyzerM2","DAL2",100,true);
  d6.Loop();
  d6.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d7("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/2012Total.root","HcalIsoTrkAnalyzer","D120",100,true);
  d7.Loop();
  d7.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d8("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/2015BCD.root","HcalIsoTrkAnalyzerM0","DV00",100,true);
  d8.Loop();
  d8.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d9("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/2015BCD.root","HcalIsoTrkAnalyzerM2","DV02",100,true);
  d9.Loop();
  d9.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n1("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/SingleMuon2015D.root","HcalIsoTrkAnalyzerM0","DMD0",100,true);
  n1.Loop();
  n1.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n2("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/SingleMuon2015D.root","HcalIsoTrkAnalyzerM2","DMD2",100,true);
  n2.Loop();
  n2.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n3("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/DoubleEG2015D.root","HcalIsoTrkAnalyzerM0","DED0",100,true);
  n3.Loop();
  n3.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n4("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/DoubleEG2015D.root","HcalIsoTrkAnalyzerM2","DED2",100,true);
  n4.Loop();
  n4.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n5("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/DoubleEGCOMB.root","HcalIsoTrkAnalyzerM0","DEG0",100,true);
  n5.Loop();
  n5.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n6("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/DoubleEGCOMB.root","HcalIsoTrkAnalyzerM2","DEG2",100,true);
  n6.Loop();
  n6.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n7("/afs/cern.ch/work/g/gwalia/public/JetHT_BCD.root","HcalIsoTrkAnalyzerM0","DJT0",100,true);
  n7.Loop();
  n7.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n8("/afs/cern.ch/work/g/gwalia/public/JetHT_BCD.root","HcalIsoTrkAnalyzerM2","DJT2",100,true);
  n8.Loop();
  n8.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n9("/afs/cern.ch/work/g/gwalia/public/JetHT_Egamma_BCD.root","HcalIsoTrkAnalyzerM0","DXG0",100,true);
  n9.Loop();
  n9.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n0("/afs/cern.ch/work/g/gwalia/public/JetHT_Egamma_BCD.root","HcalIsoTrkAnalyzerM2","DXG2",100,true);
  n0.Loop();
  n0.SavePlot(outfile,true);
}

void doPlotN(std::string outfile="histodatan.root") {
  IsoTrkOfflineAnalyzer n5("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/DoubleEGCOMB.root","HcalIsoTrkAnalyzerM0","DEG0",20100,true);
  n5.Loop();
  n5.SavePlot(outfile,false);
  IsoTrkOfflineAnalyzer n6("/afs/cern.ch/work/s/shghosh/public/MYRESULTS/DoubleEGCOMB.root","HcalIsoTrkAnalyzerM2","DEG2",20100,true);
  n6.Loop();
  n6.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n7("/afs/cern.ch/work/g/gwalia/public/JetHT_BCD.root","HcalIsoTrkAnalyzerM0","DJT0",10100,true);
  n7.Loop();
  n7.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n8("/afs/cern.ch/work/g/gwalia/public/JetHT_BCD.root","HcalIsoTrkAnalyzerM2","DJT2",10100,true);
  n8.Loop();
  n8.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n9("/afs/cern.ch/work/g/gwalia/public/JetHT_Egamma_BCD.root","HcalIsoTrkAnalyzerM0","DXG0",10100,true);
  n9.Loop();
  n9.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer n0("/afs/cern.ch/work/g/gwalia/public/JetHT_Egamma_BCD.root","HcalIsoTrkAnalyzerM2","DXG2",10100,true);
  n0.Loop();
  n0.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d8("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/2015BCD.root","HcalIsoTrkAnalyzerM0","DV00",10100,true);
  d8.Loop();
  d8.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer d9("/afs/cern.ch/work/s/sunanda/public/CMSSW_7_6_0_pre4/src/2015BCD.root","HcalIsoTrkAnalyzerM2","DV02",10100,true);
  d9.Loop();
  d9.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m7("/afs/cern.ch/work/g/gwalia/public/output_pi50_25nsBX_method2_753_2711.root", "method0", "P750",20200,false);
  m7.Loop();
  m7.SavePlot(outfile,true);
  IsoTrkOfflineAnalyzer m8("/afs/cern.ch/work/g/gwalia/public/output_pi50_25nsBX_method2_753_2711.root", "method2", "P752",20200,false);
  m8.Loop();
  m8.SavePlot(outfile,true);
}

void doFit(std::string infile="histodata.root", 
	   std::string outfile="histfitnew3.root") {
  FitHists(infile,outfile,"D250",1111,false);
  FitHists(infile,outfile,"D252",1111,true);
  FitHists(infile,outfile,"D500",1111,true);
  FitHists(infile,outfile,"D502",1111,true);
  FitHists(infile,outfile,"DAL0",1111,true);
  FitHists(infile,outfile,"DAL2",1111,true);
  FitHists(infile,outfile,"D120",1111,true);
  FitHists(infile,outfile,"DV00",1111,true,true);
  FitHists(infile,outfile,"DV02",1111,true,true);
  FitHists(infile,outfile,"DJT0",1111,true,true);
  FitHists(infile,outfile,"DJT2",1111,true,true);
  FitHists(infile,outfile,"DEG0",1111,true,true);
  FitHists(infile,outfile,"DEG2",1111,true,true);
  FitHists(infile,outfile,"DV10",1111,true);
  FitHists(infile,outfile,"DV12",1111,true);
  FitHists(infile,outfile,"DV20",1111,true);
  FitHists(infile,outfile,"DV22",1111,true);
  FitHists(infile,outfile,"DV30",1111,true);
  FitHists(infile,outfile,"DV32",1111,true);
  FitHists(infile,outfile,"DNC0",1111,true);
  FitHists(infile,outfile,"DNC2",1111,true);
  FitHists(infile,outfile,"DHR0",1111,true);
  FitHists(infile,outfile,"DHR2",1111,true);
  FitHists(infile,outfile,"DPR0",1111,true);
  FitHists(infile,outfile,"DPR2",1111,true);
  FitHists(infile,outfile,"D750",1111,true);
  FitHists(infile,outfile,"D752",1111,true);
  FitHists(infile,outfile,"D5B0",1111,true);
  FitHists(infile,outfile,"D5B2",1111,true);
  FitHists(infile,outfile,"D5C0",1111,true);
  FitHists(infile,outfile,"D5C2",1111,true);
  FitHists(infile,outfile,"D5D0",1111,true);
  FitHists(infile,outfile,"D5D2",1111,true);
  FitHists(infile,outfile,"DED0",1111,true);
  FitHists(infile,outfile,"DED2",1111,true);
  FitHists(infile,outfile,"DMD0",1111,true);
  FitHists(infile,outfile,"DMD2",1111,true);
  FitHists(infile,outfile,"Q250",1111,true);
  FitHists(infile,outfile,"Q500",1111,true);
  FitHists(infile,outfile,"PNP0",1111,true);
  FitHists(infile,outfile,"PNP2",1111,true);
  FitHists(infile,outfile,"PNP3",1111,true);
  FitHists(infile,outfile,"P250",1111,true);
  FitHists(infile,outfile,"P252",1111,true);
  FitHists(infile,outfile,"P253",1111,true);
  FitHists(infile,outfile,"P750",1111,true,true);
  FitHists(infile,outfile,"P752",1111,true,true);
}

void doFitN(std::string infile="histodatan.root", 
	    std::string outfile="histfitnew4.root") {
  FitCombineHist(infile,outfile,"DEG0","DJT0","DV00",false);
  FitCombineHist(infile,outfile,"DEG2","DJT2","DV02",true);
  FitCombineHist(infile,outfile,"P750","XXX0","P250",true);
  FitCombineHist(infile,outfile,"P752","XXX2","P252",true);
}

void doGetEntry() {
  GetEntries m1("/afs/cern.ch/work/g/gwalia/public/output_pi50_25nsBX_method2_753_2711.root", "method2", false);
  m1.Loop();
}
