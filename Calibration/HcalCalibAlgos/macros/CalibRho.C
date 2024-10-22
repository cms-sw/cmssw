//////////////////////////////////////////////////////////////////////////////
//
// Usage:
// .L CalibRho.C+g
//
//  EHcalVsRho c1(inpFilName, dupFileName);
//  c1.LoopFill(maxEta, outFile1, logFile);       Fills E vs rho plots
//  FitEvsRho(logFile, parameterFile, outFile2);  Fits the area vs iEta
//  c1.LoopTest(maxEta, parameterFile, rootFile); Makes the corrected histos
//  FitEovPwithRho(maxEta, rootFile, outFile3);   Fits E/p plots
//  PlotEvsRho(inFile, eta, type, save);          Make the plots
//
//  inpFileName   (const char*)  File name of the input ROOT tree
//                               or name of the file containing a list of
//                               file names of input ROOT trees
//  dupFileName   (const char*)  Name of the file containing list of entries
//                               of duplicate events
//  maxEta        (int)          Maximum value of |iEta|
//  outFile1      (const char*)  Output ROOT file name which will contain the
//                               scatter plots and profile histograms which
//                               will provide the estimate of "effective area"
//  logFile       (const char*)  Name of the text file which will contain the
//                               effective area for each ieta value
//  parameterFile (const char*)  Name of the text file with values of the
//                               fitted parameter set
//  outFile2      (const char*)  Name of the ROOT file with the results of
//                               the fit
//  rootFile      (const char*)  Name of the ROOT file with corrected and
//                               uncorrected histograms of E/p
//  outFile3      (const char*)  Name of the ROOT file containing the E/p
//                               histograms and the fit results
//  eta           (int)          Plot the histograms for the given iEta value
//                               (if it is 0; plots for all iEta's)
//  type          (int)          Plot E vs Rho (type=0) or E/p fits (type=1)
//  save          (bool)         Save the plots as pdf file
//////////////////////////////////////////////////////////////////////////////

#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TH1.h>
#include <TH2.h>
#include <TMultiGraph.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TStyle.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

#include "CalibCorr.C"

class EHcalVsRho {
public:
  TChain *fChain;  //!pointer to the analyzed TTree or TChain
  Int_t fCurrent;  //!current Tree number in a TChain

  EHcalVsRho(const char *inFile, const char *dupFile);
  virtual ~EHcalVsRho();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TChain *tree, const char *dupFile);
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);
  void LoopFill(int maxEta, const char *outFile, const char *logFile);
  void LoopTest(int maxEta, const char *inFile, const char *outFile);
  double EffCalc(TH1D *, double, double &, double &);
  double getEA(const int ieta, const double *par);

private:
  // Declaration of leaf types
  Int_t t_Run;
  Int_t t_Event;
  Int_t t_DataType;
  Int_t t_ieta;
  Int_t t_iphi;
  Double_t t_EventWeight;
  Int_t t_nVtx;
  Int_t t_nTrk;
  Int_t t_goodPV;
  Double_t t_l1pt;
  Double_t t_l1eta;
  Double_t t_l1phi;
  Double_t t_l3pt;
  Double_t t_l3eta;
  Double_t t_l3phi;
  Double_t t_p;
  Double_t t_pt;
  Double_t t_phi;
  Double_t t_mindR1;
  Double_t t_mindR2;
  Double_t t_eMipDR;
  Double_t t_eHcal;
  Double_t t_eHcal10;
  Double_t t_eHcal30;
  Double_t t_hmaxNearP;
  Double_t t_rhoh;
  Bool_t t_selectTk;
  Bool_t t_qltyFlag;
  Bool_t t_qltyMissFlag;
  Bool_t t_qltyPVFlag;
  Double_t t_gentrackP;
  std::vector<unsigned int> *t_DetIds;
  std::vector<double> *t_HitEnergies;
  std::vector<bool> *t_trgbits;
  std::vector<unsigned int> *t_DetIds1;
  std::vector<unsigned int> *t_DetIds3;
  std::vector<double> *t_HitEnergies1;
  std::vector<double> *t_HitEnergies3;

  // List of branches
  TBranch *b_t_Run;           //!
  TBranch *b_t_Event;         //!
  TBranch *b_t_DataType;      //!
  TBranch *b_t_ieta;          //!
  TBranch *b_t_iphi;          //!
  TBranch *b_t_EventWeight;   //!
  TBranch *b_t_nVtx;          //!
  TBranch *b_t_nTrk;          //!
  TBranch *b_t_goodPV;        //!
  TBranch *b_t_l1pt;          //!
  TBranch *b_t_l1eta;         //!
  TBranch *b_t_l1phi;         //!
  TBranch *b_t_l3pt;          //!
  TBranch *b_t_l3eta;         //!
  TBranch *b_t_l3phi;         //!
  TBranch *b_t_p;             //!
  TBranch *b_t_pt;            //!
  TBranch *b_t_phi;           //!
  TBranch *b_t_mindR1;        //!
  TBranch *b_t_mindR2;        //!
  TBranch *b_t_eMipDR;        //!
  TBranch *b_t_eHcal;         //!
  TBranch *b_t_eHcal10;       //!
  TBranch *b_t_eHcal30;       //!
  TBranch *b_t_hmaxNearP;     //!
  TBranch *b_t_rhoh;          //!
  TBranch *b_t_selectTk;      //!
  TBranch *b_t_qltyFlag;      //!
  TBranch *b_t_qltyMissFlag;  //!
  TBranch *b_t_qltyPVFlag;    //!
  TBranch *b_t_gentrackP;     //!
  TBranch *b_t_DetIds;        //!
  TBranch *b_t_HitEnergies;   //!
  TBranch *b_t_trgbits;       //!
  TBranch *b_t_DetIds1;       //!
  TBranch *b_t_DetIds3;       //!
  TBranch *b_t_HitEnergies1;  //!
  TBranch *b_t_HitEnergies3;  //!

  std::vector<Long64_t> entries_;
};

EHcalVsRho::EHcalVsRho(const char *inFile, const char *dupFile) : fChain(0) {
  char treeName[400];
  sprintf(treeName, "HcalIsoTrkAnalyzer/CalibTree");
  TChain *chain = new TChain(treeName);
  std::cout << "Create a chain for " << treeName << " from " << inFile << std::endl;
  if (!fillChain(chain, inFile)) {
    std::cout << "*****No valid tree chain can be obtained*****" << std::endl;
  } else {
    std::cout << "Proceed with a tree chain with " << chain->GetEntries() << " entries" << std::endl;
    Init(chain, dupFile);
  }
}

EHcalVsRho::~EHcalVsRho() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t EHcalVsRho::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t EHcalVsRho::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain)
    return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0)
    return centry;
  if (fChain->GetTreeNumber() != fCurrent) {
    fCurrent = fChain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void EHcalVsRho::Init(TChain *tree, const char *dupFile) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  t_DetIds = 0;
  t_HitEnergies = 0;
  t_trgbits = 0;
  t_DetIds1 = 0;
  t_DetIds3 = 0;
  t_HitEnergies1 = 0;
  t_HitEnergies3 = 0;
  // Set branch addresses and branch pointers
  if (!tree)
    return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_DataType", &t_DataType, &b_t_DataType);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_iphi", &t_iphi, &b_t_iphi);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_nVtx", &t_nVtx, &b_t_nVtx);
  fChain->SetBranchAddress("t_nTrk", &t_nTrk, &b_t_nTrk);
  fChain->SetBranchAddress("t_goodPV", &t_goodPV, &b_t_goodPV);
  fChain->SetBranchAddress("t_l1pt", &t_l1pt, &b_t_l1pt);
  fChain->SetBranchAddress("t_l1eta", &t_l1eta, &b_t_l1eta);
  fChain->SetBranchAddress("t_l1phi", &t_l1phi, &b_t_l1phi);
  fChain->SetBranchAddress("t_l3pt", &t_l3pt, &b_t_l3pt);
  fChain->SetBranchAddress("t_l3eta", &t_l3eta, &b_t_l3eta);
  fChain->SetBranchAddress("t_l3phi", &t_l3phi, &b_t_l3phi);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_pt", &t_pt, &b_t_pt);
  fChain->SetBranchAddress("t_phi", &t_phi, &b_t_phi);
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_eHcal10", &t_eHcal10, &b_t_eHcal10);
  fChain->SetBranchAddress("t_eHcal30", &t_eHcal30, &b_t_eHcal30);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_rhoh", &t_rhoh, &b_t_rhoh);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyFlag", &t_qltyFlag, &b_t_qltyFlag);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
  fChain->SetBranchAddress("t_gentrackP", &t_gentrackP, &b_t_gentrackP);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  fChain->SetBranchAddress("t_trgbits", &t_trgbits, &b_t_trgbits);
  fChain->SetBranchAddress("t_DetIds1", &t_DetIds1, &b_t_DetIds1);
  fChain->SetBranchAddress("t_DetIds3", &t_DetIds3, &b_t_DetIds3);
  fChain->SetBranchAddress("t_HitEnergies1", &t_HitEnergies1, &b_t_HitEnergies1);
  fChain->SetBranchAddress("t_HitEnergies3", &t_HitEnergies3, &b_t_HitEnergies3);
  Notify();

  if (std::string(dupFile) != "") {
    ifstream infile(dupFile);
    if (!infile.is_open()) {
      std::cout << "Cannot open " << dupFile << std::endl;
    } else {
      while (1) {
        Long64_t jentry;
        infile >> jentry;
        if (!infile.good())
          break;
        entries_.push_back(jentry);
      }
      infile.close();
      std::cout << "Reads a list of " << entries_.size() << " events from " << dupFile << std::endl;
    }
  }
}

Bool_t EHcalVsRho::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void EHcalVsRho::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t EHcalVsRho::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void EHcalVsRho::LoopFill(int maxEta, const char *outFile, const char *logFile) {
  if (fChain == 0)
    return;
  TFile *f1 = new TFile(outFile, "RECREATE");
  char name[100], Title[100], graph[100], proji[100];

  std::vector<TH2D *> VIsoRho;
  std::vector<TProfile *> Hcal_corr;
  for (int ieta = 0; ieta < maxEta; ieta++) {
    sprintf(name, "IsoRho2d%d", ieta + 1);
    sprintf(Title, "Iso vs Rho %d", ieta + 1);
    VIsoRho.push_back(new TH2D(name, Title, 30, 0, 30, 25000, 0, 250));
    sprintf(name, "IsoRhoProfile%d", ieta + 1);
    Hcal_corr.push_back(new TProfile(name, Title, 30, 0, 30));
  }

  Long64_t nentries = fChain->GetEntriesFast();
  std::cout << "Total # of entries: " << nentries << std::endl;
  Long64_t nbytes = 0, nb = 0;
  Long64_t kount(0), duplicate(0), good(0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;

    ++kount;
    if (kount % 100000 == 0)
      std::cout << "Processing Entry " << kount << std::endl;
    bool select = (std::find(entries_.begin(), entries_.end(), jentry) == entries_.end());
    if (!select) {
      ++duplicate;
      continue;
    }

    int absIeta = abs(t_ieta);
    if ((absIeta <= maxEta) && (t_p >= 40) && (t_p <= 60)) {
      VIsoRho[absIeta - 1]->Fill(t_rhoh, t_eHcal);
      Hcal_corr[absIeta - 1]->Fill(t_rhoh, t_eHcal);
      ++good;
    }
  }
  std::cout << "Uses " << good << " events out of " << kount << " excluding " << duplicate << " duplicate events"
            << std::endl;

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(1);
  std::ofstream myfile;
  myfile.open(logFile);
  for (int ieta = 0; ieta < maxEta; ieta++) {
    VIsoRho[ieta]->Write();
    Hcal_corr[ieta]->Write();
    TH2D *his_i = dynamic_cast<TH2D *>(VIsoRho[ieta]->Clone());
    his_i->GetEntries();
    int dim = his_i->GetXaxis()->GetNbins();
    double *xcut, *binc, *errXL, *errXH, *errYL, *errYH;
    double *errX, *errY;
    double errX1, errX2, xmax(0);

    xcut = new double[dim];
    binc = new double[dim];
    errX = new double[dim];
    errY = new double[dim];
    errXL = new double[dim];
    errXH = new double[dim];
    errYL = new double[dim];
    errYH = new double[dim];

    for (int j = 0; j < dim; j++) {
      sprintf(proji, "proj%d-%d", ieta + 1, j);
      TH1D *h_proj = dynamic_cast<TH1D *>(his_i->ProjectionY(proji, j, j + 1, " "));
      binc[j] = his_i->GetXaxis()->GetBinCenter(j + 1);
      xcut[j] = EffCalc(h_proj, 0.90, errX1, errX2);

      errXL[j] = 0.0;
      errXH[j] = 0.0;
      errYL[j] = errX1;
      errYH[j] = errX2;

      errX[j] = 0.0;
      errY[j] = 0.0;
      h_proj->Write();
      if (xcut[j] > xmax)
        xmax = xcut[j];
    }

    TGraphAsymmErrors *Isovsrho = new TGraphAsymmErrors(dim, binc, xcut, errXL, errXH, errYL, errYH);
    sprintf(graph, "IsovsRho%d", ieta + 1);
    sprintf(name, "EvsRho%d", ieta + 1);

    TF1 *fnc = new TF1("fnc", "[1]*x + [0]", 4, 13);
    TFitResultPtr fitI = Isovsrho->Fit("fnc", "+QSR");
    double ic = fnc->GetParameter(1);
    double err = fitI->FitResult::Error(1);
    myfile << ieta + 1 << " " << ic << " " << err << std::endl;
    std::cout << "Fit " << ieta + 1 << " " << fnc->GetParameter(0) << " " << fitI->FitResult::Error(0) << " " << ic
              << " " << err << "\n";
    gStyle->SetOptFit(1);
    TCanvas *pad = new TCanvas(graph, name, 0, 10, 1200, 400);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    Isovsrho->SetMarkerStyle(24);
    Isovsrho->SetMarkerSize(0.4);
    Isovsrho->GetXaxis()->SetRangeUser(0, 15);
    Isovsrho->GetXaxis()->SetTitle("#rho");
    Isovsrho->GetXaxis()->SetLabelSize(0.04);
    Isovsrho->GetXaxis()->SetTitleSize(0.06);
    Isovsrho->GetXaxis()->SetTitleOffset(0.8);
    Isovsrho->GetYaxis()->SetRangeUser(0, 1.25 * xmax);
    Isovsrho->GetYaxis()->SetTitle("Energy (GeV)");
    Isovsrho->GetYaxis()->SetLabelSize(0.04);
    Isovsrho->GetYaxis()->SetTitleSize(0.06);
    Isovsrho->GetYaxis()->SetTitleOffset(0.6);
    Isovsrho->Draw("AP");
    pad->Update();
    TPaveStats *st1 = (TPaveStats *)Isovsrho->GetListOfFunctions()->FindObject("stats");
    if (st1 != nullptr) {
      st1->SetY1NDC(0.78);
      st1->SetY2NDC(0.90);
      st1->SetX1NDC(0.65);
      st1->SetX2NDC(0.90);
    }
    pad->Write();
  }

  myfile.close();
  f1->Close();
}

double EHcalVsRho::EffCalc(TH1D *h, double perc, double &errXL, double &errXH) {
  double eff, eff_err = 0.0, xCut = 0.0;
  int tot = h->GetEntries();
  int integ = 0;
  errXL = 0.0;
  errXH = 0.0;
  for (int i = 0; i < (h->GetXaxis()->GetNbins() + 1); i++) {
    xCut = h->GetXaxis()->GetBinLowEdge(i);
    integ += h->GetBinContent(i);

    if (integ != 0 && tot != 0) {
      eff = (integ * 1.0 / tot);
      eff_err = sqrt(eff * (1 - eff) / tot);
    } else {
      eff = 0.0;
    }
    if (eff > perc)
      break;
  }
  if (eff == 0.0)
    xCut = 0.0;
  errXL = eff_err;
  errXH = eff_err;
  return xCut;
}

void EHcalVsRho::LoopTest(int maxEta, const char *inFile, const char *outFile) {
  if (fChain == 0)
    return;

  TFile *f1 = new TFile(outFile, "RECREATE");
  std::map<int, TH1D *> histo, histo_uncorr;
  char name[100], title[100];
  for (int ieta = -maxEta; ieta <= maxEta; ieta++) {
    sprintf(name, "MPV%d", ieta);
    sprintf(title, "Corrected Response (i#eta = %d)", ieta - 30);
    histo[ieta] = new TH1D(name, title, 100, 0, 2);
    sprintf(name, "MPVUn%d", ieta);
    sprintf(title, "Uncorrected Response (i#eta = %d)", ieta - 30);
    histo_uncorr[ieta] = new TH1D(name, title, 100, 0, 2);
  }
  std::cout << "Initialized histograms from " << -maxEta << ":" << maxEta << "\n";

  double par[10];
  ifstream myReadFile;
  myReadFile.open(inFile);
  int npar = 0;
  if (myReadFile.is_open()) {
    while (!myReadFile.eof()) {
      myReadFile >> par[npar];
      ++npar;
    }
  }
  myReadFile.close();
  std::cout << "Reads " << npar << " parameters:";
  for (int k = 0; k < npar; ++k)
    std::cout << " [" << k << "] = " << par[k];
  std::cout << std::endl;

  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;
  Long64_t kount(0), duplicate(0), good(0);
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    ++kount;
    if (kount % 100000 == 0)
      std::cout << "Processing Entry " << kount << std::endl;
    bool select = (std::find(entries_.begin(), entries_.end(), jentry) == entries_.end());
    if (!select) {
      ++duplicate;
      continue;
    }

    select = ((t_qltyFlag) && (t_selectTk) && (t_hmaxNearP < 10.0) && (t_eMipDR < 1.0) && (t_p > 40) && (t_p < 60.0));
    if (select) {
      double corr_eHcal = 0.0;
      int absIeta = abs(t_ieta);
      ++good;
      if (absIeta <= maxEta) {
        corr_eHcal = t_eHcal - t_rhoh * getEA(absIeta, par);
        double myEovP = corr_eHcal / (t_p - t_eMipDR);
        double myEovP_uncorr = t_eHcal / (t_p - t_eMipDR);
        histo[t_ieta]->Fill(myEovP);
        histo_uncorr[t_ieta]->Fill(myEovP_uncorr);
      }
    }
  }

  for (std::map<int, TH1D *>::iterator itr = histo.begin(); itr != histo.end(); ++itr)
    itr->second->Write();
  for (std::map<int, TH1D *>::iterator itr = histo_uncorr.begin(); itr != histo_uncorr.end(); ++itr)
    itr->second->Write();
  f1->Close();
  std::cout << "Processes " << good << " out of " << kount << " events with " << duplicate << " duplicate entries"
            << std::endl;
}

double EHcalVsRho::getEA(const int eta, const double *par) {
  double eA;
  if (eta < 20)
    eA = par[0];
  else
    eA = (((par[5] * eta + par[4]) * eta + par[3]) * eta + par[2]) * eta + par[1];
  return eA;
}

void FitEvsRho(const char *inFile, const char *outFile, const char *rootFile) {
  const int ndim = 30;
  double EA[ndim] = {0.0};
  double errEA[ndim] = {0.0};
  double ietaEA[ndim] = {0.0};
  ifstream myReadFile;
  myReadFile.open(inFile);

  int ii = 0;
  if (myReadFile.is_open()) {
    while (!myReadFile.eof()) {
      myReadFile >> ietaEA[ii] >> EA[ii] >> errEA[ii];
      if (EA[ii] < 0)
        EA[ii] = 0;
      ii++;
    }
  }
  myReadFile.close();
  std::cout << "Reads " << ii << " points from " << inFile << std::endl;

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
  TFile *f1 = new TFile(rootFile, "RECREATE");
  TGraphErrors *eA = new TGraphErrors(ii, ietaEA, EA, errEA, errEA);
  eA->SetMarkerStyle(20);
  eA->SetMarkerColor(4);
  eA->SetLineColor(2);
  eA->GetXaxis()->SetTitle("i#eta");
  eA->GetXaxis()->SetTitleOffset(0.6);
  eA->GetXaxis()->SetTitleSize(0.06);
  eA->GetYaxis()->SetTitle("Effective Area");
  eA->GetYaxis()->SetTitleOffset(0.6);
  eA->GetYaxis()->SetTitleSize(0.06);

  Double_t par[6];
  const int nmid = 19;
  TF1 *g1 = new TF1("g1", "pol0", 1, nmid);
  TF1 *g2 = new TF1("g2", "pol4", nmid, ii);

  eA->Fit(g1, "R");
  eA->Fit(g2, "R+");
  g1->GetParameters(&par[0]);
  g2->GetParameters(&par[1]);

  TCanvas *c2 = new TCanvas("EA vs #eta", "EA vs ieta", 0, 10, 1200, 400);
  eA->Draw("AP");
  c2->Write();
  f1->Close();

  ofstream params;
  params.open(outFile);
  for (int i = 0; i < 6; i++) {
    params << par[i] << std::endl;
    std::cout << "Parameter[" << i << "] = " << par[i] << std::endl;
  }
  params.close();
}

void FitEovPwithRho(int maxEta, const char *inFile, const char *outFile) {
  TFile *file = new TFile(inFile);
  std::map<int, TH1D *> histo, histo_uncorr;
  char name[100];
  for (int ieta = -maxEta; ieta <= maxEta; ieta++) {
    sprintf(name, "MPV%d", ieta);
    TH1D *h0 = (TH1D *)file->FindObjectAny(name);
    histo[ieta] = (h0 != 0) ? (TH1D *)(h0->Clone()) : 0;
    sprintf(name, "MPVUn%d", ieta);
    TH1D *h1 = (TH1D *)file->FindObjectAny(name);
    histo_uncorr[ieta] = (h1 != 0) ? (TH1D *)(h1->Clone()) : 0;
  }

  //TFile *f1 =
  new TFile(outFile, "RECREATE");
  double xlim = maxEta + 0.5;
  TH1D *EovPvsieta = new TH1D("Corrected", "Corrected", 2 * maxEta + 1, -xlim, xlim);
  TH1D *EovPvsieta_uncorr = new TH1D("Uncorrect", "Uncorrect", 2 * maxEta + 1, -xlim, xlim);

  TF1 *fnc = new TF1("fnc", "gaus");
  unsigned int k1(0), k2(0);
  for (int ieta = -maxEta; ieta <= maxEta; ieta++) {
    if (ieta == 0)
      continue;
    if (histo[ieta] != 0) {
      double mean = histo[ieta]->GetMean();
      double rms = histo[ieta]->GetRMS();
      TFitResultPtr FitG = histo[ieta]->Fit("fnc", "QRWLS", "", mean - rms, mean + rms);
      double a = fnc->GetParameter(1);
      double err = FitG->FitResult::Error(1);
      histo[ieta]->Write();
      ++k1;
      int ibin = ieta + maxEta + 1;
      EovPvsieta->SetBinContent(ibin, a);
      EovPvsieta->SetBinError(ibin, err);
      std::cout << "Correct[" << k1 << "] " << ieta << " a " << a << " +- " << err << std::endl;
    }

    if (histo_uncorr[ieta] != 0) {
      double mean = histo_uncorr[ieta]->GetMean();
      double rms = histo_uncorr[ieta]->GetRMS();
      TFitResultPtr FitG = histo_uncorr[ieta]->Fit("fnc", "QRWLS", "", mean - rms, mean + rms);
      double a = fnc->GetParameter(1);
      double err = FitG->FitResult::Error(1);
      histo_uncorr[ieta]->Write();
      ++k2;
      int ibin = ieta + maxEta + 1;
      EovPvsieta_uncorr->SetBinContent(ibin, a);
      EovPvsieta_uncorr->SetBinError(ibin, err);
      std::cout << "Correct[" << k2 << "] " << ieta << " a " << a << " +- " << err << std::endl;
    }
  }

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(10);
  gStyle->SetOptFit(1);
  TCanvas *c3 = new TCanvas("E/P vs ieta", "E/P vs ieta", 0, 10, 1200, 400);
  EovPvsieta->GetXaxis()->SetTitle("i#eta");
  EovPvsieta->GetYaxis()->SetTitle("MPV[E_{Hcal}/(p_{Track}-E_{Ecal})]");
  EovPvsieta->SetMarkerStyle(20);
  EovPvsieta->SetMarkerColor(2);
  EovPvsieta->SetMarkerSize(1.0);
  EovPvsieta->Fit("pol0", "+QRWLS", "", -maxEta, maxEta);

  EovPvsieta_uncorr->SetMarkerStyle(24);
  EovPvsieta_uncorr->SetMarkerColor(4);
  EovPvsieta_uncorr->SetMarkerSize(1.0);

  EovPvsieta->GetYaxis()->SetRangeUser(0.5, 2.0);
  EovPvsieta->Draw();
  c3->Update();
  TPaveStats *st1 = (TPaveStats *)EovPvsieta->GetListOfFunctions()->FindObject("stats");
  if (st1 != nullptr) {
    st1->SetY1NDC(0.81);
    st1->SetY2NDC(0.90);
    st1->SetX1NDC(0.65);
    st1->SetX2NDC(0.90);
  }
  EovPvsieta_uncorr->Draw("sames");
  c3->Update();
  st1 = (TPaveStats *)EovPvsieta_uncorr->GetListOfFunctions()->FindObject("stats");
  std::cout << st1 << std::endl;
  if (st1 != nullptr) {
    st1->SetY1NDC(0.78);
    st1->SetY2NDC(0.81);
    st1->SetX1NDC(0.65);
    st1->SetX2NDC(0.90);
  }
  c3->Modified();
  c3->Update();
  EovPvsieta->Write();
  EovPvsieta_uncorr->Write();
}

void PlotEvsRho(const char *inFile, int eta = 0, int type = 0, bool save = false) {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(1110);
  gStyle->SetOptFit(10);

  TFile *file = new TFile(inFile);
  int etamin = (eta == 0) ? ((type == 0) ? 1 : 25) : eta;
  int etamax = (eta == 0) ? 25 : eta;
  for (int it = etamin; it <= etamax; ++it) {
    char name[50];
    sprintf(name, "IsovsRho%d", it);
    TCanvas *pad;
    if (type == 0) {
      pad = (TCanvas *)(file->FindObjectAny(name));
      pad->Draw();
    } else {
      sprintf(name, "MPV%d", it);
      pad = new TCanvas(name, name, 0, 10, 800, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      TH1D *h1 = (TH1D *)(file->FindObjectAny(name));
      sprintf(name, "MPVUn%d", it);
      TH1D *h2 = (TH1D *)(file->FindObjectAny(name));
      double ymx1 = h1->GetMaximum();
      double ymx2 = h2->GetMaximum();
      double ymax = (ymx1 > ymx2) ? ymx1 : ymx2;
      h1->GetXaxis()->SetRangeUser(0.5, 2.0);
      h2->GetXaxis()->SetRangeUser(0.5, 2.0);
      h1->GetYaxis()->SetRangeUser(0, 1.25 * ymax);
      h2->GetYaxis()->SetRangeUser(0, 1.25 * ymax);
      h1->GetXaxis()->SetTitleSize(0.048);
      h1->GetXaxis()->SetTitleOffset(0.8);
      h1->GetXaxis()->SetTitle("E_{Hcal}/(p-E_{Ecal})");
      h1->GetYaxis()->SetTitleSize(0.048);
      h1->GetYaxis()->SetTitleOffset(0.8);
      h1->GetYaxis()->SetTitle("Tracks");
      h1->SetLineColor(2);
      h1->Draw();
      pad->Modified();
      pad->Update();
      TPaveStats *st1 = (TPaveStats *)h1->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        st1->SetLineColor(2);
        st1->SetTextColor(2);
        st1->SetY1NDC(0.75);
        st1->SetY2NDC(0.90);
        st1->SetX1NDC(0.65);
        st1->SetX2NDC(0.90);
      }
      h2->SetLineColor(4);
      h2->Draw("sames");
      pad->Modified();
      pad->Update();
      TPaveStats *st2 = (TPaveStats *)h2->GetListOfFunctions()->FindObject("stats");
      if (st2 != nullptr) {
        st2->SetLineColor(4);
        st2->SetTextColor(4);
        st2->SetY1NDC(0.60);
        st2->SetY2NDC(0.75);
        st2->SetX1NDC(0.65);
        st2->SetX2NDC(0.90);
      }
      pad->Modified();
      pad->Update();
      TF1 *f1 = (TF1 *)h1->GetListOfFunctions()->FindObject("fnc");
      if (f1 != nullptr)
        f1->SetLineColor(2);
      TF1 *f2 = (TF1 *)h2->GetListOfFunctions()->FindObject("fnc");
      if (f2 != nullptr)
        f2->SetLineColor(4);
      pad->Modified();
      pad->Update();
    }
    if (save) {
      sprintf(name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}
