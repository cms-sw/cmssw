//////////////////////////////////////////////////////////////////////////////
// This class analyzes the "Lep Tree" created by HBHEMuonOfflineAnalyzer
// It has two constructors using either a pointer to the tree chain or
// the file name where the tree resides
// There are 2 additional arguments:
//      mode (a packed integer) with bits specifying some action
//           Bit 0 : 0 will produce set of histograms of energy deposit,
//                     active length, active length corrected energy;
//                   1 will produce plots of p, nvx and scatter plots for
//                     each ieta
//               1 : 0 ignores the information on number of vertex
//                   1 groups ranges of # of vertex 0:15:20:25:30:100
//               2 : 0 ignores the information on track momentum
//                   1 separate plots for certain momentum range
//                     (the range depends on ieta)
//              3-4: 0 separate plot for each depth
//                   1 sums up all depths
//                   2 collapse depth
//              5-6: 0 no check on iphi
//                   1 separate plot for each iphi
//                   2 separate plot for each RBX
//                   3 exclude the RBX specified by bits 6-10
//             7-11:   RBX # to be excluded (maximum 5 bits needed for RBX)
//               12: 0 varying ranges of p depending on |ieta|
//                   1 constant p ranges
//            13-15: 0 no cut on ediff; 1-4 cuts at 5, 10, 15, 20 GeV
//      modeLHC (integer) specifies the detector condition
//              1 (Run 1; valid till 2016); 2 (Run 2; 2018);
//              3 (Run 3; post LS2); 4 (2017 Plan 1);
//              5 (Run 4; post LS3); (default: 3)
//
//   AnalyzeLepTree a1(tree, mode, modeLHC);
//   AnalyzeLepTree a1(fname, mode, modeLHC);
//   a1.Loop();
//   a1.writeHisto(outfile);
//   a1.plotHisto(drawStatBox,type,save);
//   a1.writeMeanError(outfileMean);
//
//      outfile     (char*)  Name of the file where histograms to be written
//      outfileMean (char*)  Name of the file where means with errors to be
//                           written when it is run with mode bit 0 set to 1
//      drawStatBox (bool)   If Statbox to be drawn or not
//      type        (int)    Each bit says what plots to be drawn
//                           If bit 0 of "mode" is 1
//                           (0: momentum for each ieta;
//                            1: number of vertex for each ieta;
//                            2: 2D plot for nvx vs p for each ieta;
//                            3: number of vertex for each ieta & depth;
//                            4: momentum for each ieta & depth
//                            5: energy in the outer ring)
//                           If bit 0 of "mode" is 0
//                           plots for all or specific depth & phi if
//                           "depth" as of (type/16)&15 is 0 or the specified
//                           value and "phi" as of (type/256)&127 is 0 or
//                           the specified value. Bit 0 set plots the energy
//                           distribution; 1 set plots active length corrected
//                           energy; 2 set plots charge; 3 set plots active
//                           length corrected charge distributions
//      save        (bool)   If plots to be saved as pdf file or not
///////////////////////////////////////////////////////////////////////////////

#include <TCanvas.h>
#include <TChain.h>
#include <TColor.h>
#include <TFile.h>
#include <TH2.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <vector>

class AnalyzeLepTree {
public:
  AnalyzeLepTree(TChain* tree, int mode = 0, int modeLHC = 3);
  AnalyzeLepTree(const char* fname, int mode = 0, int modeLHC = 3);
  virtual ~AnalyzeLepTree();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain* tree);
  virtual void Loop(Long64_t nmax = -1, bool debug = false);
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
  void writeHisto(const char* outfile);
  void writeMeanError(const char* outfile);
  std::vector<TCanvas*> plotHisto(bool drawStatBox, int type, bool save = false);

private:
  bool fillChain(TChain* chain, const char* fname);
  void bookHisto();
  void plotHisto(std::map<unsigned int, TH1D*> hists, std::vector<TCanvas*>& cvs, bool save);
  void plotHisto(std::map<unsigned int, TH1D*> hists, int phi0, int depth0, std::vector<TCanvas*>& cvs, bool save);
  TCanvas* plotHisto(TH1D* hist);
  void plot2DHisto(std::map<unsigned int, TH2D*> hists, std::vector<TCanvas*>& cvs, bool save);
  int getCollapsedDepth(int eta, int phi, int depth);
  int getRBX(int eta);
  int getPBin(int eta);
  int getVxBin();
  int getDepthBin(int depth);
  int getPhiBin(int eta);
  void makeVxBins(int modeLHC);
  int nDepthBins(int eta, int phi, int modeLHC);
  int nPhiBins(int eta);
  int nPBins(int eta);
  int nVxBins();
  unsigned int packID(int zside, int eta, int phi, int depth, int nvx, int ipbin);
  void unpackID(unsigned int id, int& zside, int& eta, int& phi, int& depth, int& nvx, int& ipbin);
  void getBins(int type, int eta, int phi, int depth, int& nbins, double& xmax);

private:
  TChain* fChain;  //!pointer to the analyzed TTree or TChain
  Int_t fCurrent;  //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t t_ieta;
  Int_t t_iphi;
  Int_t t_nvtx;
  Double_t t_p;
  Double_t t_ediff;
  std::vector<double>* t_ene;
  std::vector<double>* t_enec;
  std::vector<double>* t_charge;
  std::vector<double>* t_actln;
  std::vector<int>* t_depth;

  // List of branches
  TBranch* b_t_ieta;    //!
  TBranch* b_t_iphi;    //!
  TBranch* b_t_nvtx;    //!
  TBranch* b_t_p;       //!
  TBranch* b_t_ediff;   //!
  TBranch* b_t_ene;     //!
  TBranch* b_t_enec;    //!
  TBranch* b_t_charge;  //!
  TBranch* b_t_actln;   //!
  TBranch* b_t_depth;   //!

  static const int etamax_ = 26, npbin_ = 9, nvbin_ = 6;
  static const bool debug_ = false;
  int mode_, modeLHC_, exRBX_, kphi_, kdepth_;
  std::vector<int> npvbin_, iprange_;
  std::vector<double> prange_[5];
  double cutEdiff_;
  std::map<unsigned int, TH1D*> h_p_, h_nv_;
  std::map<unsigned int, TH2D*> h_pnv_;
  std::map<unsigned int, TH1D*> h_p2_, h_nv2_;
  std::map<unsigned int, TH1D*> h_Energy_, h_Ecorr_, h_Charge_, h_Chcorr_;
  std::map<unsigned int, TH1D*> h_EnergyC_, h_EcorrC_;
  std::map<unsigned int, TH1D*> h_ediff_, h_ediff_nvtx_;
};

AnalyzeLepTree::AnalyzeLepTree(TChain* tree, int mode1, int mode2) : mode_(mode1), modeLHC_(mode2) {
  std::cout << "Proceed with a tree chain with " << tree->GetEntries() << " entries" << std::endl;
  Init(tree);
}

AnalyzeLepTree::AnalyzeLepTree(const char* fname, int mode1, int mode2) : mode_(mode1), modeLHC_(mode2) {
  TChain* chain = new TChain("Lep_Tree");
  if (!fillChain(chain, fname)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain);
  }
}

AnalyzeLepTree::~AnalyzeLepTree() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t AnalyzeLepTree::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t AnalyzeLepTree::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain)
    return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0)
    return centry;
  if (!fChain->InheritsFrom(TChain::Class()))
    return centry;
  TChain* chain = (TChain*)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void AnalyzeLepTree::Init(TChain* tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  makeVxBins(modeLHC_);
  exRBX_ = (mode_ / 128) % 32;
  kphi_ = (mode_ / 32) % 4;
  kdepth_ = (mode_ / 8) % 4;
  if ((mode_ % 2) == 0)
    std::cout << "Produce set of histograms of energy, "
              << " active length, active length corrected "
              << "energy for 3 types" << std::endl;
  else
    std::cout << "Produce plots of p, nvx and scatter plots "
              << "for each ieta" << std::endl;
  if (((mode_ / 2) % 2) == 0) {
    std::cout << "Ignore the information on number of vertex iformation" << std::endl;
  } else {
    std::cout << "Group ranges of # of vertex ";
    for (unsigned int k = 0; k < npvbin_.size(); ++k)
      std::cout << npvbin_[k] << ":";
    std::cout << std::endl;
  }
  if (((mode_ / 4) % 2) == 0)
    std::cout << "Ignore the information on track "
              << "momentum" << std::endl;
  else
    std::cout << "Separate plots for certain momentum "
              << "range (the range depends on ieta)\n";
  if (kdepth_ == 0)
    std::cout << "Generate separate plot for each depth" << std::endl;
  else if (kdepth_ == 1)
    std::cout << "Sums up all depths for plots\n";
  else
    std::cout << "Collapse depths to Run 1 scenario\n";
  if (kphi_ == 0)
    std::cout << "Make no check on iphi" << std::endl;
  else if (kphi_ == 1)
    std::cout << "Make separate plot for each iphi\n";
  else if (kphi_ == 2)
    std::cout << "Make separate plot for each RBX\n";
  else
    std::cout << "Exclude the RBX " << exRBX_ << std::endl;
  if (modeLHC_ == 1)
    std::cout << "This is Run1 detector (till 2016)\n";
  else if (modeLHC_ == 2)
    std::cout << "This is Plan36 detector (2018)\n";
  else if (modeLHC_ == 3)
    std::cout << "This is Phase1 detector (after 2021)\n";
  else if (modeLHC_ == 4)
    std::cout << "This is Plan1  detector (2017)\n";
  else
    std::cout << "This is Phase2 detector (after 2026)\n";
  static const double cuts[8] = {200, 5, 10, 15, 20, 25, 30, 40};
  int cutE = (mode_ / 4096) % 8;
  cutEdiff_ = cuts[cutE];
  std::cout << "Cut off for energy in the 8 neighbouring towers " << cutEdiff_ << std::endl;

  // Set object pointer
  t_ene = 0;
  t_enec = 0;
  t_charge = 0;
  t_actln = 0;
  t_depth = 0;
  fChain = tree;
  // Set branch addresses and branch pointers
  if (!tree)
    return;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_iphi", &t_iphi, &b_t_iphi);
  fChain->SetBranchAddress("t_nvtx", &t_nvtx, &b_t_nvtx);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_ediff", &t_ediff, &b_t_ediff);
  fChain->SetBranchAddress("t_ene", &t_ene, &b_t_ene);
  fChain->SetBranchAddress("t_enec", &t_enec, &b_t_enec);
  fChain->SetBranchAddress("t_charge", &t_charge, &b_t_charge);
  fChain->SetBranchAddress("t_actln", &t_actln, &b_t_actln);
  fChain->SetBranchAddress("t_depth", &t_depth, &b_t_depth);
  Notify();

  t_ediff = 0;

  bookHisto();
}

Bool_t AnalyzeLepTree::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  return kTRUE;
}

void AnalyzeLepTree::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t AnalyzeLepTree::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void AnalyzeLepTree::Loop(Long64_t nmax, bool debug) {
  //   In a ROOT session, you can do:
  //      Root > .L AnalyzeLepTree.C
  //      Root > AnalyzeLepTree t
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
  if (fChain == 0)
    return;

  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "Number of entries: " << nentries << ":" << nmax << std::endl;
  if (nmax > 0 && nmax < nentries)
    nentries = nmax;
  const double ethr = 0.00001;  // Threshold of 10 keV

  Long64_t nbytes = 0, nb = 0;
  int32_t n15(0), n16(0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if ((jentry % 1000000 == 0) || debug)
      std::cout << "Entry " << jentry << ":" << ientry << std::endl;
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    // if (Cut(ientry) < 0) continue;
    int zside = (t_ieta > 0) ? 1 : -1;
    int eta = (t_ieta > 0) ? t_ieta : -t_ieta;
    int phi = getPhiBin(eta);
    int pbin = getPBin(eta);
    int vbin = getVxBin();
    if ((mode_ / 1) % 2 == 1) {
      unsigned int id0 = packID(zside, eta, 1, 1, 1, 1);
      std::map<unsigned int, TH1D*>::iterator it1 = h_p_.find(id0);
      if (it1 != h_p_.end())
        (it1->second)->Fill(t_p);
      std::map<unsigned int, TH1D*>::iterator it2 = h_nv_.find(id0);
      if (it2 != h_nv_.end())
        (it2->second)->Fill(t_nvtx);
      std::map<unsigned int, TH2D*>::iterator it3 = h_pnv_.find(id0);
      if (it3 != h_pnv_.end())
        (it3->second)->Fill(t_nvtx, t_p);
      unsigned int id1 = packID(zside, eta, 1, 1, 1, pbin);
      std::map<unsigned int, TH1D*>::iterator it4 = h_p2_.find(id1);
      if (it4 != h_p2_.end())
        (it4->second)->Fill(t_p);
      unsigned int id2 = packID(zside, eta, 1, 1, vbin, 1);
      std::map<unsigned int, TH1D*>::iterator it5 = h_nv2_.find(id2);
      if (it5 != h_nv2_.end())
        (it5->second)->Fill(t_nvtx);
      std::map<unsigned int, TH1D*>::iterator it6 = h_ediff_.find(id0);
      if (it6 != h_ediff_.end())
        (it6->second)->Fill(t_ediff);
      std::map<unsigned int, TH1D*>::iterator it7 = h_ediff_nvtx_.find(id2);
      if (it7 != h_ediff_nvtx_.end())
        (it7->second)->Fill(t_ediff);
    } else {
      if (phi > 0 && pbin >= 0 && vbin >= 0) {
        if (kdepth_ == 0) {
          for (unsigned int k = 0; k < t_depth->size(); ++k) {
            if (eta == 15)
              ++n15;
            else if (eta == 16)
              ++n16;
            int depth = (*t_depth)[k];
            unsigned int id = packID(zside, eta, phi, depth + 1, vbin, pbin);
            double ene = (*t_ene)[k];
            double enec = (*t_enec)[k];
            double charge = (*t_charge)[k];
            double actl = (*t_actln)[k];
            if (ene > ethr && actl > 0 && charge > 0 && t_ediff < cutEdiff_) {
              std::map<unsigned int, TH1D*>::iterator it1 = h_Energy_.find(id);
              if (it1 != h_Energy_.end())
                (it1->second)->Fill(ene);
              std::map<unsigned int, TH1D*>::iterator it2 = h_Ecorr_.find(id);
              if (it2 != h_Ecorr_.end())
                (it2->second)->Fill(ene / actl);
              std::map<unsigned int, TH1D*>::iterator it3 = h_EnergyC_.find(id);
              if (it3 != h_EnergyC_.end())
                (it3->second)->Fill(enec);
              std::map<unsigned int, TH1D*>::iterator it4 = h_EcorrC_.find(id);
              if (it4 != h_EcorrC_.end())
                (it4->second)->Fill(enec / actl);
              std::map<unsigned int, TH1D*>::iterator it5 = h_Charge_.find(id);
              if (it5 != h_Charge_.end())
                (it5->second)->Fill(charge);
              std::map<unsigned int, TH1D*>::iterator it6 = h_Chcorr_.find(id);
              if (it6 != h_Chcorr_.end())
                (it6->second)->Fill(charge / actl);
              if (debug_) {
                //		if ((eta>20 && (t_iphi > 35)) || (t_iphi > 71)) std::cout << zside << ":" << eta << ":" << phi << ":" << t_iphi << ":" << depth+1 << ":" << vbin << ":" << pbin << " ID " << std::hex << id << std::dec << " Flags " <<  (it1 != h_Energy_.end()) << ":" << (it2 != h_Ecorr_.end()) << ":" <<  (it3 != h_EnergyC_.end()) << ":" << (it4 != h_EcorrC_.end()) << ":" << (it5 != h_Charge_.end()) << ":" << (it6 != h_Chcorr_.end()) << " E " << ene << " C " << charge << " L " << actl << std::endl;
                if ((it1 == h_Energy_.end()) || (it2 == h_Ecorr_.end()) || (it3 == h_EnergyC_.end()) ||
                    (it4 == h_EcorrC_.end()) || (it5 == h_Charge_.end()) || (it6 == h_Chcorr_.end()))
                  std::cout << zside << ":" << eta << ":" << phi << ":" << t_iphi << ":" << depth + 1 << ":" << vbin
                            << ":" << pbin << " ID " << std::hex << id << std::dec << " Flags "
                            << (it1 != h_Energy_.end()) << ":" << (it2 != h_Ecorr_.end()) << ":"
                            << (it3 != h_Charge_.end()) << ":" << (it4 != h_Chcorr_.end()) << " E " << ene << " C "
                            << charge << " L " << actl << std::endl;
              }
            }
          }
        } else if (kdepth_ == 1) {
          double ene[2], enec[2], actl[2], charge[2];
          for (unsigned int k = 0; k < 2; ++k) {
            ene[k] = enec[k] = actl[k] = charge[k] = 0;
          }
          for (unsigned int k = 0; k < t_depth->size(); ++k) {
            if ((*t_ene)[k] > 0 && (*t_actln)[k] > 0) {
              int dep = (*t_depth)[k];
              int depth = ((eta != 16) ? 0 : ((dep > 1) ? 1 : 0));
              ene[depth] += (*t_ene)[k];
              enec[depth] += (*t_enec)[k];
              charge[depth] += (*t_charge)[k];
              actl[depth] += (*t_actln)[k];
            }
          }
          int nDepth = (eta == 16) ? 2 : 1;
          for (int k = 0; k < nDepth; ++k) {
            if (ene[k] > ethr && actl[k] > 0 && charge[k] > 0 && t_ediff < cutEdiff_) {
              if (eta == 15)
                ++n15;
              else if (eta == 16)
                ++n16;
              int depth = k + 1;
              unsigned int id = packID(zside, eta, phi, depth, vbin, pbin);
              std::map<unsigned int, TH1D*>::iterator it1 = h_Energy_.find(id);
              if (it1 != h_Energy_.end())
                (it1->second)->Fill(ene[k]);
              std::map<unsigned int, TH1D*>::iterator it2 = h_Ecorr_.find(id);
              if (it2 != h_Ecorr_.end())
                (it2->second)->Fill(ene[k] / actl[k]);
              std::map<unsigned int, TH1D*>::iterator it3 = h_EnergyC_.find(id);
              if (it3 != h_EnergyC_.end())
                (it3->second)->Fill(enec[k]);
              std::map<unsigned int, TH1D*>::iterator it4 = h_EcorrC_.find(id);
              if (it4 != h_EcorrC_.end())
                (it4->second)->Fill(enec[k] / actl[k]);
              std::map<unsigned int, TH1D*>::iterator it5 = h_Charge_.find(id);
              if (it5 != h_Charge_.end())
                (it5->second)->Fill(charge[k]);
              std::map<unsigned int, TH1D*>::iterator it6 = h_Chcorr_.find(id);
              if (it6 != h_Chcorr_.end())
                (it6->second)->Fill(charge[k] / actl[k]);
              if (((eta == 15) || (eta == 16)) && debug_)
                std::cout << zside << ":" << eta << ":" << phi << ":" << t_iphi << ":" << depth << ":" << vbin << ":"
                          << pbin << " ID " << std::hex << id << std::dec << " Flags " << (it1 != h_Energy_.end())
                          << ":" << (it2 != h_Ecorr_.end()) << ":" << (it3 != h_Charge_.end()) << ":"
                          << (it4 != h_Chcorr_.end()) << " E " << ene << " C " << charge << " L " << actl << std::endl;
            }
          }
        } else {
          double ene[3], enec[3], actl[3], charge[3];
          for (unsigned int k = 0; k < 3; ++k) {
            ene[k] = enec[k] = actl[k] = charge[k] = 0;
          }
          for (unsigned int k = 0; k < t_depth->size(); ++k) {
            if ((*t_ene)[k] > 0 && (*t_actln)[k] > 0) {
              int dep = (*t_depth)[k];
              int depth = getCollapsedDepth(zside * eta, phi, dep) - 1;
              ene[depth] += (*t_ene)[k];
              enec[depth] += (*t_enec)[k];
              charge[depth] += (*t_charge)[k];
              actl[depth] += (*t_actln)[k];
            }
          }
          for (int k = 0; k < nDepthBins(eta, phi, 0); ++k) {
            if (ene[k] > ethr && actl[k] > 0 && charge[k] > 0 && t_ediff < cutEdiff_) {
              if (eta == 15)
                ++n15;
              else if (eta == 16)
                ++n16;
              int depth = k + 1;
              unsigned int id = packID(zside, eta, phi, depth, vbin, pbin);
              std::map<unsigned int, TH1D*>::iterator it1 = h_Energy_.find(id);
              if (it1 != h_Energy_.end())
                (it1->second)->Fill(ene[k]);
              std::map<unsigned int, TH1D*>::iterator it2 = h_Ecorr_.find(id);
              if (it2 != h_Ecorr_.end())
                (it2->second)->Fill(ene[k] / actl[k]);
              std::map<unsigned int, TH1D*>::iterator it3 = h_EnergyC_.find(id);
              if (it3 != h_EnergyC_.end())
                (it3->second)->Fill(enec[k]);
              std::map<unsigned int, TH1D*>::iterator it4 = h_EcorrC_.find(id);
              if (it4 != h_EcorrC_.end())
                (it4->second)->Fill(enec[k] / actl[k]);
              std::map<unsigned int, TH1D*>::iterator it5 = h_Charge_.find(id);
              if (it5 != h_Charge_.end())
                (it5->second)->Fill(charge[k]);
              std::map<unsigned int, TH1D*>::iterator it6 = h_Chcorr_.find(id);
              if (it6 != h_Chcorr_.end())
                (it6->second)->Fill(charge[k] / actl[k]);
              if (((eta == 15) || (eta == 16)) && debug_)
                std::cout << zside << ":" << eta << ":" << phi << ":" << t_iphi << ":" << depth << ":" << vbin << ":"
                          << pbin << " ID " << std::hex << id << std::dec << " Flags " << (it1 != h_Energy_.end())
                          << ":" << (it2 != h_Ecorr_.end()) << ":" << (it3 != h_Charge_.end()) << ":"
                          << (it4 != h_Chcorr_.end()) << " E " << ene[k] << " C " << charge[k] << " L " << actl[k]
                          << std::endl;
            }
          }
        }
      }
    }
  }
  std::cout << "Number of events with eta15: " << n15 << " and eta16: " << n16 << std::endl;
}

bool AnalyzeLepTree::fillChain(TChain* chain, const char* inputFileList) {
  int kount(0);
  std::string fname(inputFileList);
  if (fname.substr(fname.size() - 5, 5) == ".root") {
    chain->Add(fname.c_str());
  } else {
    ifstream infile(inputFileList);
    if (!infile.is_open()) {
      std::cout << "** ERROR: Can't open '" << inputFileList << "' for input" << std::endl;
      return false;
    }
    while (1) {
      infile >> fname;
      if (!infile.good())
        break;
      chain->Add(fname.c_str());
      ++kount;
    }
    infile.close();
  }
  std::cout << "Adds " << kount << " files in the chain from " << fname << std::endl;
  return true;
}

void AnalyzeLepTree::bookHisto() {
  for (int i = 0; i < 5; ++i)
    prange_[i].clear();
  int ipbrng[30] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4};
  for (int i = 0; i < 30; ++i)
    iprange_.push_back(ipbrng[i]);
  double prange0[npbin_] = {0, 30, 45, 55, 75, 100, 125, 150, 500};
  double prange1[npbin_] = {0, 50, 75, 100, 125, 150, 200, 300, 500};
  double prange2[npbin_] = {0, 60, 75, 100, 125, 150, 200, 300, 500};
  double prange3[npbin_] = {0, 100, 125, 150, 175, 200, 300, 400, 500};
  double prange4[npbin_] = {0, 125, 150, 175, 200, 250, 300, 400, 500};
  double prangeX[npbin_] = {125, 150, 200, 250, 300, 350, 400, 500};
  for (int i = 0; i < npbin_; ++i) {
    if ((mode_ / 4096) % 2 == 0) {
      prange_[0].push_back(prange0[i]);
      prange_[1].push_back(prange1[i]);
      prange_[2].push_back(prange2[i]);
      prange_[3].push_back(prange3[i]);
      prange_[4].push_back(prange4[i]);
    } else {
      prange_[0].push_back(prangeX[i]);
      prange_[1].push_back(prangeX[i]);
      prange_[2].push_back(prangeX[i]);
      prange_[3].push_back(prangeX[i]);
      prange_[4].push_back(prangeX[i]);
    }
  }
  if (debug_) {
    std::cout << "Eta range " << -etamax_ << ":" << etamax_ << " # of vtx bins " << nVxBins() << std::endl;
    if ((mode_ / 1) % 2 == 0) {
      for (int ieta = -etamax_; ieta <= etamax_; ++ieta) {
        int eta = (ieta > 0) ? ieta : -ieta;
        if (eta != 0) {
          int ndepth = ((kdepth_ == 0) ? nDepthBins(eta, 63, modeLHC_)
                                       : ((kdepth_ != 1) ? nDepthBins(eta, 63, 0)
                                          : (eta == 16)  ? 2
                                                         : 1));
          std::cout << "Eta " << ieta << " with " << nPhiBins(eta) << " phi bins " << ndepth << " maximum depths and "
                    << nPBins(eta) << " p bins" << std::endl;
        }
      }
    }
  }

  char name[100], title[200];
  unsigned int book1(0), book2(0);
  if ((mode_ / 1) % 2 == 1) {
    h_p_.clear();
    h_nv_.clear();
    h_pnv_.clear();
    h_nv2_.clear();
    h_p2_.clear();
    h_ediff_.clear();
    h_ediff_nvtx_.clear();
    for (int ieta = -etamax_; ieta <= etamax_; ++ieta) {
      if (ieta != 0) {
        int zside = (ieta > 0) ? 1 : -1;
        int eta = (ieta > 0) ? ieta : -ieta;
        unsigned int id0 = packID(zside, eta, 1, 1, 1, 1);
        sprintf(name, "peta%d", ieta);
        sprintf(title, "Momentum for i#eta = %d (GeV)", ieta);
        h_p_[id0] = new TH1D(name, title, 100, 0.0, 500.0);
        ++book1;
        sprintf(name, "Ediff_eta%d", ieta);
        sprintf(title, "Energy difference for i#eta = %d (GeV)", ieta);
        h_ediff_[id0] = new TH1D(name, title, 1000, -500.0, 500.0);
        ++book1;
        sprintf(name, "nveta%d", ieta);
        sprintf(title, "Number of Vertex for i#eta = %d", ieta);
        h_nv_[id0] = new TH1D(name, title, 100, 0.0, 100.0);
        ++book1;
        sprintf(name, "pnveta%d", ieta);
        sprintf(title, "i#eta = %d", ieta);
        TH2D* h2 = new TH2D(name, title, 100, 0.0, 100.0, 100, 0.0, 500.0);
        ++book2;
        h2->GetXaxis()->SetTitle("Number of Vertex");
        h2->GetYaxis()->SetTitle("Momentum (GeV)");
        h_pnv_[id0] = h2;
        ++book1;
        char etas[10];
        sprintf(etas, "i#eta=%d", ieta);
        char name[100], title[200];
        for (int pbin = 0; pbin < nPBins(eta); ++pbin) {
          char ps[20];
          if ((mode_ / 4) % 2 == 1) {
            int np = (eta >= 0 && eta < (int)(iprange_.size())) ? iprange_[eta] - 1 : iprange_[0];
            sprintf(ps, "p=%d:%d", (int)prange_[np][pbin], (int)prange_[np][pbin + 1]);
          };
          unsigned int id = packID(zside, eta, 1, 1, 1, pbin);
          sprintf(name, "pvx%d111%d", ieta, pbin);
          sprintf(title, "Momentum for %s %s", etas, ps);
          h_p2_[id] = new TH1D(name, title, 100, 0.0, 500.0);
          h_p2_[id]->Sumw2();
          ++book1;
        }
        for (int vbin = 0; vbin < nVxBins(); ++vbin) {
          char vtx[24];
          if ((mode_ / 2) % 2 == 1) {
            sprintf(vtx, "N_{vtx}=%d:%d", npvbin_[vbin], npvbin_[vbin + 1]);
          } else {
            sprintf(vtx, "all N_{vtx}");
          }
          unsigned int id = packID(zside, eta, 1, 1, vbin, 1);
          sprintf(name, "nvx%d11%d1", ieta, vbin);
          sprintf(title, "Number of vertex for %s %s", etas, vtx);
          h_nv2_[id] = new TH1D(name, title, 100, 0.0, 100.0);
          h_nv2_[id]->Sumw2();
          ++book1;
          sprintf(name, "Ediff_nvx%d11%d1", ieta, vbin);
          sprintf(title, "Energy difference for %s %s", etas, vtx);
          h_ediff_nvtx_[id] = new TH1D(name, title, 1000, -500.0, 500.0);
          h_ediff_nvtx_[id]->Sumw2();
          ++book1;
        }
      }
    }
  } else {
    h_Energy_.clear();
    h_Ecorr_.clear();
    h_Charge_.clear();
    h_Chcorr_.clear();
    h_EnergyC_.clear();
    h_EcorrC_.clear();
    for (int ieta = -etamax_; ieta <= etamax_; ++ieta) {
      if (ieta != 0) {
        int zside = (ieta > 0) ? 1 : -1;
        int eta = (ieta > 0) ? ieta : -ieta;
        char etas[20];
        sprintf(etas, "i#eta=%d", ieta);
        for (int iphi = 0; iphi < nPhiBins(eta); ++iphi) {
          char phis[20];
          int phi(1), phi0(63);
          if (kphi_ == 1) {
            phi0 = phi = (eta <= 20) ? iphi + 1 : 2 * iphi + 1;
            sprintf(phis, "i#phi=%d", phi);
          } else if (kphi_ == 2) {
            phi0 = 4 * iphi + 1;
            phi = iphi + 1;
            sprintf(phis, "RBX=%d", iphi + 1);
          } else if (kphi_ == 3) {
            sprintf(phis, "All except RBX %d", exRBX_);
          } else {
            sprintf(phis, "All i#phi");
          }
          int ndepth = ((kdepth_ == 0) ? nDepthBins(eta, phi0, modeLHC_)
                                       : ((kdepth_ != 1) ? nDepthBins(eta, phi0, 0)
                                          : (eta == 16)  ? 2
                                                         : 1));
          for (int depth = 0; depth < ndepth; ++depth) {
            char deps[20];
            if (kdepth_ == 1) {
              if (depth == 0)
                sprintf(deps, "all depths");
              else
                sprintf(deps, "all endcap depths");
            } else {
              sprintf(deps, "Depth=%d", depth + 1);
            }
            for (int pbin = 0; pbin < nPBins(eta); ++pbin) {
              char ps[30];
              if ((mode_ / 4) % 2 == 1) {
                int np = (eta >= 0 && eta < (int)(iprange_.size())) ? iprange_[eta] - 1 : iprange_[0];
                sprintf(ps, "p=%d:%d", (int)prange_[np][pbin], (int)prange_[np][pbin + 1]);
              } else {
                sprintf(ps, "all p");
              };
              for (int vbin = 0; vbin < nVxBins(); ++vbin) {
                int nbin(4000);
                double xmax(10.0);
                char vtx[20];
                if ((mode_ / 2) % 2 == 1) {
                  sprintf(vtx, "N_{vtx}=%d:%d", npvbin_[vbin], npvbin_[vbin + 1]);
                } else {
                  sprintf(vtx, "all N_{vtx}");
                }
                unsigned int id = packID(zside, eta, phi, depth + 1, vbin, pbin);
                char name[100], title[200];
                sprintf(name, "EdepE%dF%dD%dV%dP%d", ieta, phi, depth, vbin, pbin);
                sprintf(title, "Deposited energy for %s %s %s %s %s (GeV)", etas, phis, deps, ps, vtx);
                getBins(0, ieta, phi0, depth + 1, nbin, xmax);
                h_Energy_[id] = new TH1D(name, title, nbin, 0.0, xmax);
                ++book1;
                sprintf(name, "EcorE%dF%dD%dV%dP%d", ieta, phi, depth, vbin, pbin);
                sprintf(title, "Active length corrected energy for %s %s %s %s %s (GeV/cm)", etas, phis, deps, ps, vtx);
                getBins(1, ieta, phi0, depth + 1, nbin, xmax);
                h_Ecorr_[id] = new TH1D(name, title, nbin, 0.0, xmax);
                ++book1;
                sprintf(name, "EdepCE%dF%dD%dV%dP%d", ieta, phi, depth, vbin, pbin);
                sprintf(
                    title, "Response Corrected deposited energy for %s %s %s %s %s (GeV)", etas, phis, deps, ps, vtx);
                getBins(2, ieta, phi0, depth + 1, nbin, xmax);
                h_EnergyC_[id] = new TH1D(name, title, nbin, 0.0, xmax);
                ++book1;
                sprintf(name, "EcorCE%dF%dD%dV%dP%d", ieta, phi, depth, vbin, pbin);
                sprintf(title,
                        "Response and active length corrected energy for %s %s %s %s %s (GeV/cm)",
                        etas,
                        phis,
                        deps,
                        ps,
                        vtx);
                getBins(3, ieta, phi0, depth + 1, nbin, xmax);
                h_EcorrC_[id] = new TH1D(name, title, nbin, 0.0, xmax);
                ++book1;
                sprintf(name, "ChrgE%dF%dD%dV%dP%d", ieta, phi, depth, vbin, pbin);
                sprintf(title, "Measured charge for %s %s %s %s %s (cm)", etas, phis, deps, ps, vtx);
                getBins(4, ieta, phi0, depth + 1, nbin, xmax);
                h_Charge_[id] = new TH1D(name, title, nbin, 0.0, xmax);
                ++book1;
                sprintf(name, "ChcorE%dF%dD%dV%dP%d", ieta, phi, depth, vbin, pbin);
                sprintf(title, "Active length corrected charge for %s %s %s %s %s (cm)", etas, phis, deps, ps, vtx);
                getBins(5, ieta, phi0, depth + 1, nbin, xmax);
                h_Chcorr_[id] = new TH1D(name, title, nbin, 0.0, xmax);
                ++book1;
              }
            }
          }
        }
      }
    }
  }
  std::cout << "Booked " << book1 << " 1D and " << book2 << " 2D Histos\n";
}

void AnalyzeLepTree::writeHisto(const char* outfile) {
  TFile* output_file = TFile::Open(outfile, "RECREATE");
  output_file->cd();
  if ((mode_ / 1) % 2 == 1) {
    for (std::map<unsigned int, TH1D*>::const_iterator itr = h_p_.begin(); itr != h_p_.end(); ++itr)
      (itr->second)->Write();
    for (std::map<unsigned int, TH1D*>::const_iterator itr = h_nv_.begin(); itr != h_nv_.end(); ++itr)
      (itr->second)->Write();
    for (std::map<unsigned int, TH2D*>::const_iterator itr = h_pnv_.begin(); itr != h_pnv_.end(); ++itr)
      (itr->second)->Write();
    for (std::map<unsigned int, TH1D*>::const_iterator itr = h_p2_.begin(); itr != h_p2_.end(); ++itr)
      (itr->second)->Write();
    for (std::map<unsigned int, TH1D*>::const_iterator itr = h_nv2_.begin(); itr != h_nv2_.end(); ++itr)
      (itr->second)->Write();
    for (std::map<unsigned int, TH1D*>::const_iterator itr = h_ediff_.begin(); itr != h_ediff_.end(); ++itr)
      (itr->second)->Write();
    for (std::map<unsigned int, TH1D*>::const_iterator itr = h_ediff_nvtx_.begin(); itr != h_ediff_nvtx_.end(); ++itr)
      (itr->second)->Write();
  } else {
    for (int ieta = -etamax_; ieta <= etamax_; ++ieta) {
      if (ieta != 0) {
        char dirname[50];
        sprintf(dirname, "DieMuonEta%d", ieta);
        TDirectory* d_output = output_file->mkdir(dirname);
        d_output->cd();
        int zside = (ieta > 0) ? 1 : -1;
        int eta = (ieta > 0) ? ieta : -ieta;
        for (int iphi = 0; iphi < nPhiBins(eta); ++iphi) {
          int phi(1), phi0(1);
          if (kphi_ == 1) {
            phi0 = phi = (eta <= 20) ? iphi + 1 : 2 * iphi + 1;
          } else if (kphi_ == 2) {
            phi0 = 4 * iphi + 1;
            phi = iphi + 1;
          };
          int ndepth = ((kdepth_ == 0) ? nDepthBins(eta, phi0, modeLHC_)
                                       : ((kdepth_ != 1) ? nDepthBins(eta, phi0, 0)
                                          : (eta == 16)  ? 2
                                                         : 1));
          for (int depth = 0; depth < ndepth; ++depth) {
            for (int pbin = 0; pbin < nPBins(eta); ++pbin) {
              for (int vbin = 0; vbin < nVxBins(); ++vbin) {
                unsigned int id = packID(zside, eta, phi, depth + 1, vbin, pbin);
                std::map<unsigned int, TH1D*>::const_iterator itr;
                itr = h_Energy_.find(id);
                if (itr != h_Energy_.end())
                  (itr->second)->Write();
                itr = h_Ecorr_.find(id);
                if (itr != h_Ecorr_.end())
                  (itr->second)->Write();
                itr = h_EnergyC_.find(id);
                if (itr != h_EnergyC_.end())
                  (itr->second)->Write();
                itr = h_EcorrC_.find(id);
                if (itr != h_EcorrC_.end())
                  (itr->second)->Write();
                itr = h_Charge_.find(id);
                if (itr != h_Charge_.end())
                  (itr->second)->Write();
                itr = h_Chcorr_.find(id);
                if (itr != h_Chcorr_.end())
                  (itr->second)->Write();
              }
            }
          }
        }
      }
      output_file->cd();
    }
  }
}

void AnalyzeLepTree::writeMeanError(const char* outfile) {
  if ((mode_ / 1) % 2 == 1) {
    std::ofstream fOutput(outfile);
    for (int vbin = 0; vbin < nVxBins(); ++vbin) {
      for (int ieta = -etamax_; ieta <= etamax_; ++ieta) {
        if (ieta != 0) {
          int zside = (ieta > 0) ? 1 : -1;
          int eta = (ieta > 0) ? ieta : -ieta;
          unsigned int id = packID(zside, eta, 1, 1, vbin, 1);
          char name[100];
          sprintf(name, "nvx%d11%d1", ieta, vbin);
          std::map<unsigned int, TH1D*>::iterator itr = h_nv2_.find(id);
          if (itr != h_nv2_.end()) {
            double mean = (itr->second)->GetMean();
            double error = (itr->second)->GetMeanError();
            char vtx[24];
            if ((mode_ / 2) % 2 == 1) {
              sprintf(vtx, "Nvtx=%3d:%3d", npvbin_[vbin], npvbin_[vbin + 1]);
            } else {
              sprintf(vtx, "all Nvtx");
            }
            fOutput << "Eta " << std::setw(3) << ieta << " " << vtx << " " << mean << " +- " << error << std::endl;
          }
        }
      }
    }
    fOutput.close();
  }
}

std::vector<TCanvas*> AnalyzeLepTree::plotHisto(bool drawStatBox, int type, bool save) {
  std::vector<TCanvas*> cvs;
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (drawStatBox) {
    gStyle->SetOptStat(111110);
    gStyle->SetOptFit(1);
  } else {
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
  }

  if ((mode_ / 1) % 2 == 1) {
    if (type % 2 > 0)
      plotHisto(h_p_, cvs, save);
    if ((type / 2) % 2 > 0)
      plotHisto(h_nv_, cvs, save);
    if ((type / 4) % 2 > 0)
      plot2DHisto(h_pnv_, cvs, save);
    if ((type / 8) % 2 > 0)
      plotHisto(h_nv2_, cvs, save);
    if ((type / 16) % 2 > 0)
      plotHisto(h_p2_, cvs, save);
    if ((type / 32) % 2 > 0)
      plotHisto(h_ediff_, cvs, save);
    if ((type / 32) % 2 > 0)
      plotHisto(h_ediff_nvtx_, cvs, save);
  } else {
    int depth0 = (type / 16) & 15;
    int phi0 = (type / 256) & 127;
    bool doEn = ((type / 1) % 2 > 0);
    bool doEnL = ((type / 2) % 2 > 0);
    bool doChg = ((type / 4) % 2 > 0);
    bool doChL = ((type / 8) % 2 > 0);
    if (doEn)
      plotHisto(h_Energy_, phi0, depth0, cvs, save);
    if (doEn)
      plotHisto(h_EnergyC_, phi0, depth0, cvs, save);
    if (doEnL)
      plotHisto(h_Ecorr_, phi0, depth0, cvs, save);
    if (doEnL)
      plotHisto(h_EcorrC_, phi0, depth0, cvs, save);
    if (doChg)
      plotHisto(h_Charge_, phi0, depth0, cvs, save);
    if (doChL)
      plotHisto(h_Chcorr_, phi0, depth0, cvs, save);
  }
  return cvs;
}

void AnalyzeLepTree::plotHisto(std::map<unsigned int, TH1D*> hists, std::vector<TCanvas*>& cvs, bool save) {
  for (std::map<unsigned int, TH1D*>::const_iterator itr = hists.begin(); itr != hists.end(); ++itr) {
    TH1D* hist = (itr->second);
    if (hist != 0) {
      TCanvas* pad = plotHisto(hist);
      cvs.push_back(pad);
      if (save) {
        char name[100];
        sprintf(name, "c_%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

void AnalyzeLepTree::plotHisto(
    std::map<unsigned int, TH1D*> hists, int phi0, int depth0, std::vector<TCanvas*>& cvs, bool save) {
  for (std::map<unsigned int, TH1D*>::const_iterator itr = hists.begin(); itr != hists.end(); ++itr) {
    int zside, eta, phi, depth, pbin, vbin;
    unpackID(itr->first, zside, eta, phi, depth, vbin, pbin);
    TH1D* hist = itr->second;
    if (((depth == depth0) || (depth0 == 0)) && ((phi == phi0) || (phi0 == 0)) && (hist != 0)) {
      TCanvas* pad = plotHisto(hist);
      cvs.push_back(pad);
      if (save) {
        char name[100];
        sprintf(name, "c_%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

TCanvas* AnalyzeLepTree::plotHisto(TH1D* hist) {
  TCanvas* pad = new TCanvas(hist->GetName(), hist->GetName(), 700, 500);
  pad->SetRightMargin(0.10);
  pad->SetTopMargin(0.10);
  hist->GetXaxis()->SetTitleSize(0.04);
  hist->GetXaxis()->SetTitle(hist->GetTitle());
  hist->GetYaxis()->SetTitle("Tracks");
  hist->GetYaxis()->SetLabelOffset(0.005);
  hist->GetYaxis()->SetTitleSize(0.04);
  hist->GetYaxis()->SetLabelSize(0.035);
  hist->GetYaxis()->SetTitleOffset(1.30);
  hist->SetMarkerStyle(20);
  hist->SetLineWidth(2);
  hist->Draw();
  pad->Update();
  TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
  if (st1 != NULL) {
    st1->SetY1NDC(0.70);
    st1->SetY2NDC(0.90);
    st1->SetX1NDC(0.65);
    st1->SetX2NDC(0.90);
  }
  pad->Modified();
  pad->Update();
  return pad;
}

void AnalyzeLepTree::plot2DHisto(std::map<unsigned int, TH2D*> hists, std::vector<TCanvas*>& cvs, bool save) {
  char name[100];
  for (std::map<unsigned int, TH2D*>::const_iterator itr = hists.begin(); itr != hists.end(); ++itr) {
    TH2D* hist = (itr->second);
    if (hist != 0) {
      TCanvas* pad = new TCanvas(hist->GetName(), hist->GetName(), 700, 700);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      hist->GetXaxis()->SetTitleSize(0.04);
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleSize(0.04);
      hist->GetYaxis()->SetLabelSize(0.035);
      hist->GetYaxis()->SetTitleOffset(1.30);
      hist->Draw("COLZ");
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != NULL) {
        st1->SetY1NDC(0.65);
        st1->SetY2NDC(0.90);
        st1->SetX1NDC(0.65);
        st1->SetX2NDC(0.90);
      }
      pad->Modified();
      pad->Update();
      cvs.push_back(pad);
      if (save) {
        sprintf(name, "c_%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

int AnalyzeLepTree::getCollapsedDepth(int eta, int phi, int dep) {
  int depth = dep + 1;
  int ieta = (eta > 0) ? eta : -eta;
  if (ieta <= 14 || ieta == 17) {
    depth = 1;
  } else if (ieta == 15) {
    if (modeLHC_ > 3) {
      if (dep > 3)
        depth = 2;
      else
        depth = 1;
    }
  } else if (ieta == 16) {
    if (modeLHC_ == 0 || (modeLHC_ == 3 && (phi < 63 || phi > 66 || eta < 0))) {
    } else {
      if (dep > 2)
        depth = 3;
    }
  } else if (ieta < 26) {
    if (modeLHC_ == 0 || (modeLHC_ == 3 && (phi < 63 || phi > 66 || eta < 0))) {
    } else {
      if (dep < 3)
        depth = 1;
      else
        depth = 2;
    }
  } else if (ieta == 26) {
    if (modeLHC_ == 0 || (modeLHC_ == 3 && (phi < 63 || phi > 66 || eta < 0))) {
    } else {
      if (dep < 4)
        depth = 1;
      else
        depth = 2;
    }
  } else {
    if (modeLHC_ == 0 || (modeLHC_ == 3 && (phi < 63 || phi > 66 || eta < 0))) {
    } else {
      if (dep < 3)
        depth = 1;
      else if (dep == 3)
        depth = 2;
      else
        depth = 3;
    }
  }
  return depth;
}

int AnalyzeLepTree::getRBX(int eta) {
  int rbx(1);
  int phi = (eta > 20) ? (2 * t_iphi + 1) : (t_iphi + 1);
  if (phi > 2 && phi < 71)
    rbx = (phi + 5) / 4;
  return rbx;
}

int AnalyzeLepTree::getPBin(int eta) {
  int bin(0);
  if ((mode_ / 4) % 2 == 1) {
    int np = (eta >= 0 && eta < (int)(iprange_.size())) ? iprange_[eta] : iprange_[0];
    for (unsigned int k = 1; k < prange_[np].size(); ++k) {
      if (t_p < prange_[np][k])
        break;
      bin = k;
    }
  }
  return bin;
}

int AnalyzeLepTree::getVxBin() {
  int bin(0);
  if ((mode_ / 2) % 2 == 1) {
    for (unsigned int k = 1; k < npvbin_.size(); ++k) {
      if (t_nvtx < npvbin_[k])
        break;
      bin = k;
    }
  }
  return bin;
}

int AnalyzeLepTree::getDepthBin(int depth) {
  int bin = (kdepth_ == 0) ? depth : 1;
  return bin;
}

int AnalyzeLepTree::getPhiBin(int eta) {
  int bin(1);
  if (kphi_ == 1) {
    bin = (eta > 20) ? (2 * t_iphi + 1) : (t_iphi + 1);
  } else if (kphi_ == 2) {
    bin = getRBX(eta);
  } else if (kphi_ == 3) {
    if (exRBX_ == getRBX(eta))
      bin = -1;
  }
  return bin;
}

void AnalyzeLepTree::makeVxBins(int modeLHC) {
  int npvbin0[nvbin_] = {0, 15, 20, 25, 30, 100};
  int npvbin1[nvbin_] = {0, 20, 25, 30, 35, 100};
  int npvbin2[nvbin_] = {0, 25, 30, 35, 40, 100};
  int npvbin3[nvbin_] = {0, 30, 40, 50, 70, 200};
  npvbin_.clear();
  for (int i = 0; i < nvbin_; ++i) {
    if (modeLHC == 3)
      npvbin_.push_back(npvbin0[i]);
    else if (modeLHC == 1)
      npvbin_.push_back(npvbin1[i]);
    else if ((modeLHC == 2) || (modeLHC == 4))
      npvbin_.push_back(npvbin2[i]);
    else
      npvbin_.push_back(npvbin3[i]);
  }
}

int AnalyzeLepTree::nDepthBins(int eta, int phi, int modeLHC) {
  // Run 1 scenario
  int nDepthR1[29] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2};
  // Run 2 scenario from 2018
  int nDepthR2[29] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 3, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 3};
  // Run 3 scenario
  int nDepthR3[29] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 3};
  // Run 4 scenario
  int nDepthR4[29] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
  // modeLHC = 1 -->      corresponds to Run 1 (valid till 2016)
  //         = 2 -->      corresponds to Run 2 (2018 geometry)
  //         = 3 -->      corresponds to Run 3 (post LS2)
  //         = 4 -->      corresponds to 2017  (Plan 1)
  //         = 5 -->      corresponds to Run4  (post LS3)
  int nbin(0);
  if (modeLHC == 1) {
    nbin = nDepthR1[eta - 1];
  } else if (modeLHC == 2) {
    nbin = nDepthR2[eta - 1];
  } else if (modeLHC == 3) {
    nbin = nDepthR3[eta - 1];
  } else if (modeLHC == 4) {
    if (phi > 0) {
      if (eta >= 16 && phi >= 63 && phi <= 66) {
        nbin = nDepthR2[eta - 1];
      } else {
        nbin = nDepthR1[eta - 1];
      }
    } else {
      if (eta >= 16) {
        nbin = (nDepthR2[eta - 1] > nDepthR1[eta - 1]) ? nDepthR2[eta - 1] : nDepthR1[eta - 1];
      } else {
        nbin = nDepthR1[eta - 1];
      }
    }
  } else {
    if (eta > 0 && eta < 30) {
      nbin = nDepthR4[eta - 1];
    } else {
      nbin = nDepthR4[28];
    }
  }
  return nbin;
}

int AnalyzeLepTree::nPhiBins(int eta) {
  int nphi = (eta <= 20) ? 72 : 36;
  if (modeLHC_ == 5 && eta > 16)
    nphi = 360;
  if (kphi_ == 0)
    nphi = 1;
  else if (kphi_ == 2)
    nphi = 18;
  else if (kphi_ == 3)
    nphi = 1;
  return nphi;
}

int AnalyzeLepTree::nPBins(int eta) {
  int bin(1);
  if ((mode_ / 4) % 2 == 1) {
    int np = (eta >= 0 && eta < (int)(iprange_.size())) ? iprange_[eta] - 1 : iprange_[0];
    bin = (int)(prange_[np].size()) - 1;
  }
  return bin;
}

int AnalyzeLepTree::nVxBins() {
  int nvx = ((mode_ / 2) % 2 == 1) ? ((int)(npvbin_.size()) - 1) : 1;
  return nvx;
}

unsigned int AnalyzeLepTree::packID(int zside, int eta, int phi, int depth, int nvx, int ipbin) {
  unsigned int id = (zside > 0) ? 1 : 0;
  id |= (((nvx & 7) << 19) | ((ipbin & 7) << 16) | ((depth & 7) << 13) | ((eta & 31) << 8) | ((phi & 127) << 1));
  return id;
}

void AnalyzeLepTree::unpackID(unsigned int id, int& zside, int& eta, int& phi, int& depth, int& nvx, int& ipbin) {
  zside = (id % 2 == 0) ? -1 : 1;
  phi = (id >> 1) & 127;
  eta = (id >> 8) & 31;
  depth = (id >> 13) & 7;
  ipbin = (id >> 16) & 7;
  nvx = (id >> 19) & 7;
}

void AnalyzeLepTree::getBins(int type, int ieta, int phi, int depth, int& nbin, double& xmax) {
  int eta = (ieta >= 0) ? ieta : -ieta;
  bool barrel = (eta < 16) || ((eta == 16) && (depth <= 2));
  bool rbx17 = (phi >= 63) && (phi <= 66) && (ieta >= 16) && (!barrel);
  nbin = 50000;
  xmax = 500.0;
  if (type >= 4) {
    if ((modeLHC_ == 1) || (((modeLHC_ == 2) || (modeLHC_ == 4)) && barrel) || ((modeLHC_ == 4) && (!rbx17))) {
      // HPD Channels
      nbin = 5000;
      xmax = 50.0;
    } else {
      // SiPM Channels
      xmax = 50000.0;
      nbin = 50000;
    }
  }
}
