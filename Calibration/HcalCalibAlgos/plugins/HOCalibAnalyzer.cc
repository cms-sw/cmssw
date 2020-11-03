//April 2015 : Removal of itrg1, itrg2, but addition of isect2, same is true in HOCalibVariables.h
// -*- C++ -*-
//
// Package:    HOCalibAnalyzer
// Class:      HOCalibAnalyzer
//
/**\class HOCalibAnalyzer HOCalibAnalyzer.cc Calibration/HOCalibAnalyzer/src/HOCalibAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>

April 2015
// Addition of these variables, ilumi (analyser), inslumi (analyser), nprim


*/
//
// Original Author:  Gobinda Majumder
//         Created:  Sat Jul  7 09:51:31 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "DataFormats/HOCalibHit/interface/HOCalibVariables.h"
#include "DataFormats/HcalCalibObjects/interface/HOCalibVariables.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TProfile.h"
#include "TPostScript.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TStyle.h"
#include "TMinuit.h"
#include "TMath.h"

#include <string>

#include <iostream>
#include <fstream>
#include <iomanip>
//#include <sstream>

//
//  Look for nearby pixel through eta, phi informations for pixel cross-talk
// 1. Look PIXEL code from (eta,phi)
// 2. Go to nearby pixel code
// 3. Come back to (eta,phi) from pixel code
// Though it works, it is a very ugly/crude way to get cross talk, need better algorithms
//

static const int mapx1[6][3] = {{1, 4, 8}, {12, 7, 3}, {5, 9, 13}, {11, 6, 2}, {16, 15, 14}, {19, 18, 17}};

static const int mapx2[6][3] = {{1, 4, 8}, {12, 7, 3}, {5, 9, 13}, {11, 6, 2}, {16, 15, 14}, {-1, -1, -1}};

static const int mapx0p[9][2] = {{3, 1}, {7, 4}, {6, 5}, {12, 8}, {0, 0}, {11, 9}, {16, 13}, {15, 14}, {19, 17}};
static const int mapx0m[9][2] = {{17, 19}, {14, 15}, {13, 16}, {9, 11}, {0, 0}, {8, 12}, {5, 6}, {4, 7}, {1, 3}};

static const int etamap[4][21] = {{-1, 0, 3, 1, 0, 2, 3, 1, 0, 2, -1, 3, 1, 2, 4, 4, 4, -1, -1, -1, -1},  //etamap2
                                  {-1, 0, 3, 1, 0, 2, 3, 1, 0, 2, -1, 3, 1, 2, 4, 4, 4, 5, 5, 5, -1},     //etamap1
                                  {-1, 0, -1, 0, 1, 2, 2, 1, 3, 5, -1, 5, 3, 6, 7, 7, 6, 8, -1, 8, -1},   //etamap0p
                                  {-1, 8, -1, 8, 7, 6, 6, 7, 5, 3, -1, 3, 5, 2, 1, 1, 2, 0, -1, 0, -1}};  //etamap0m

static const int phimap[4][21] = {{-1, 0, 2, 2, 1, 0, 1, 1, 2, 1, -1, 0, 0, 2, 2, 1, 0, 2, 1, 0, -1},     //phimap2
                                  {-1, 0, 2, 2, 1, 0, 1, 1, 2, 1, -1, 0, 0, 2, 2, 1, 0, 2, 1, 0, -1},     //phimap1
                                  {-1, 1, -1, 0, 1, 1, 0, 0, 1, 1, -1, 0, 0, 1, 1, 0, 0, 1, -1, 0, -1},   //phimap0p
                                  {-1, 0, -1, 1, 0, 0, 1, 1, 0, 0, -1, 1, 1, 0, 0, 1, 1, 0, -1, 1, -1}};  //phimap0m
//swapped phi map for R0+/R0- (15/03/07)

static const int npixleft[21] = {0, 0, 1, 2, 0, 4, 5, 6, 0, 8, 0, 0, 11, 0, 13, 14, 15, 0, 17, 18, 0};
static const int npixrigh[21] = {0, 2, 3, 0, 5, 6, 7, 0, 9, 0, 0, 12, 0, 14, 15, 16, 0, 18, 19, 0, 0};
static const int npixlebt[21] = {0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 0, 6, 7, 8, 9, 0, 11, 13, 14, 15, 0};
static const int npixribt[21] = {0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 0, 7, 0, 9, 0, 11, 12, 14, 15, 16, 0};
static const int npixleup[21] = {0, 4, 5, 6, 8, 9, 0, 11, 0, 13, 0, 15, 16, 0, 17, 18, 19, 0, 0, 0, 0};
static const int npixriup[21] = {0, 5, 6, 7, 9, 0, 11, 12, 13, 14, 0, 16, 0, 17, 18, 19, 0, 0, 0, 0, 0};

static const int netamx = 30;
static const int nphimx = 72;
static const int nbgpr = 3;
static const int nsgpr = 7;

int ietafit;
int iphifit;
std::vector<float> sig_reg[netamx][nphimx + 1];
std::vector<float> cro_ssg[netamx][nphimx + 1];

//#define CORREL
//
// class decleration
//

Double_t gausX(Double_t* x, Double_t* par) { return par[0] * (TMath::Gaus(x[0], par[1], par[2], kTRUE)); }

Double_t langaufun(Double_t* x, Double_t* par) {
  //Fit parameters:
  //par[0]*par[1]=Width (scale) parameter of Landau density
  //par[1]=Most Probable (MP, location) parameter of Landau density
  //par[2]=Total area (integral -inf to inf, normalization constant)
  //par[3]=Width (sigma) of convoluted Gaussian function
  //
  //In the Landau distribution (represented by the CERNLIB approximation),
  //the maximum is located at x=-0.22278298 with the location parameter=0.
  //This shift is corrected within this function, so that the actual
  //maximum is identical to the MP parameter.
  // /*
  // Numeric constants
  Double_t invsq2pi = 0.3989422804014;  // (2 pi)^(-1/2)
  Double_t mpshift = -0.22278298;       // Landau maximum location

  // Control constants
  Double_t np = 100.0;  // number of convolution steps
  Double_t sc = 5.0;    // convolution extends to +-sc Gaussian sigmas

  // Variables
  Double_t xx;
  Double_t mpc;
  Double_t fland;
  Double_t sum = 0.0;
  Double_t xlow, xupp;
  Double_t step;

  // MP shift correction
  mpc = par[1] - mpshift * par[0] * par[1];

  // Range of convolution integral
  xlow = x[0] - sc * par[3];
  xupp = x[0] + sc * par[3];

  step = (xupp - xlow) / np;

  // Convolution integral of Landau and Gaussian by sum
  for (double ij = 1.0; ij <= np / 2; ij++) {
    xx = xlow + (ij - .5) * step;
    fland = TMath::Landau(xx, mpc, par[0] * par[1], kTRUE);  // / par[0];
    sum += fland * TMath::Gaus(x[0], xx, par[3]);
    xx = xupp - (ij - .5) * step;
    fland = TMath::Landau(xx, mpc, par[0] * par[1], kTRUE);  // / par[0];
    sum += fland * TMath::Gaus(x[0], xx, par[3]);
  }

  return (par[2] * step * sum * invsq2pi / par[3]);
}

Double_t totalfunc(Double_t* x, Double_t* par) { return gausX(x, par) + langaufun(x, &par[3]); }

void fcnbg(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag) {
  double fval = -par[0];
  for (unsigned ij = 0; ij < cro_ssg[ietafit][iphifit].size(); ij++) {
    double xval = (double)cro_ssg[ietafit][iphifit][ij];
    fval += std::log(std::max(1.e-30, par[0] * TMath::Gaus(xval, par[1], par[2], true)));
    //    fval +=std::log(par[0]*TMath::Gaus(xval, par[1], par[2], 1));
  }
  f = -fval;
}

void fcnsg(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag) {
  double xval[2];
  double fval = -(par[0] + par[5]);
  for (unsigned ij = 0; ij < sig_reg[ietafit][iphifit].size(); ij++) {
    xval[0] = (double)sig_reg[ietafit][iphifit][ij];
    fval += std::log(totalfunc(xval, par));
  }
  f = -fval;
}

void set_mean(double& x, bool mdigi) {
  if (mdigi) {
    x = std::min(x, 0.5);
    x = std::max(x, -0.5);
  } else {
    x = std::min(x, 0.1);
    x = std::max(x, -0.1);
  }
}

void set_sigma(double& x, bool mdigi) {
  if (mdigi) {
    x = std::min(x, 1.2);
    x = std::max(x, -1.2);
  } else {
    x = std::min(x, 0.24);
    x = std::max(x, 0.03);
  }
}

class HOCalibAnalyzer : public edm::EDAnalyzer {
public:
  explicit HOCalibAnalyzer(const edm::ParameterSet&);
  ~HOCalibAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  int getHOieta(int ij) { return (ij < netamx / 2) ? -netamx / 2 + ij : -netamx / 2 + ij + 1; }
  int invert_HOieta(int ieta) { return (ieta < 0) ? netamx / 2 + ieta : netamx / 2 + ieta - 1; }

  TFile* theFile;
  std::string theRootFileName;
  std::string theoutputtxtFile;
  std::string theoutputpsFile;

  bool m_allHOsignal;
  bool m_hotime;
  bool m_hbtime;
  bool m_correl;
  bool m_checkmap;
  bool m_hbinfo;
  bool m_combined;
  bool m_constant;
  bool m_figure;
  bool m_cosmic;
  bool m_histfit;
  bool m_pedsuppr;
  double m_sigma;

  static const int ncut = 13;
  static const int mypow_2_ncut = 8192;  // 2^13, should be changed to match ncut

  int ipass;

  TTree* T1;

  TH2F* ho_entry;
  TH2F* ho_energy;
  TH2F* ho_energy2;
  TH2F* ho_rms;
  TH1F* ho_indenergy[netamx][nphimx];

  TH1F* muonnm;
  TH1F* muonmm;
  TH1F* muonth;
  TH1F* muonph;
  TH1F* muonch;

  TH1F* sel_muonnm;
  TH1F* sel_muonmm;
  TH1F* sel_muonth;
  TH1F* sel_muonph;
  TH1F* sel_muonch;

  TProfile* hotime[netamx][nphimx];
  TProfile* hopedtime[netamx][nphimx];
  TProfile* hbtime[netamx][nphimx];

  TH1F* corrsglb[netamx][nphimx];
  TH1F* corrsgrb[netamx][nphimx];
  TH1F* corrsglu[netamx][nphimx];
  TH1F* corrsgru[netamx][nphimx];
  TH1F* corrsgall[netamx][nphimx];

  TH1F* corrsgl[netamx][nphimx];
  TH1F* corrsgr[netamx][nphimx];

  TH1F* mncorrsglb;
  TH1F* mncorrsgrb;
  TH1F* mncorrsglu;
  TH1F* mncorrsgru;
  TH1F* mncorrsgall;

  TH1F* mncorrsgl;
  TH1F* mncorrsgr;

  TH1F* rmscorrsglb;
  TH1F* rmscorrsgrb;
  TH1F* rmscorrsglu;
  TH1F* rmscorrsgru;
  TH1F* rmscorrsgall;

  TH1F* rmscorrsgl;
  TH1F* rmscorrsgr;

  TH1F* nevcorrsglb;
  TH1F* nevcorrsgrb;
  TH1F* nevcorrsglu;
  TH1F* nevcorrsgru;
  TH1F* nevcorrsgall;

  TH1F* nevcorrsgl;
  TH1F* nevcorrsgr;

  TH1F* corrsgc[netamx][nphimx];
  TH1F* mncorrsgc;
  TH1F* rmscorrsgc;
  TH1F* nevcorrsgc;

  TH1F* sigrsg[netamx][nphimx + 1];
  TH1F* crossg[netamx][nphimx + 1];
  float invang[netamx][nphimx + 1];

  TH1F* mnsigrsg;
  TH1F* mncrossg;

  TH1F* rmssigrsg;
  TH1F* rmscrossg;

  TH1F* nevsigrsg;
  TH1F* nevcrossg;

  TH1F* ho_sig2p[9];
  TH1F* ho_sig1p[9];
  TH1F* ho_sig00[9];
  TH1F* ho_sig1m[9];
  TH1F* ho_sig2m[9];

  TH1F* hbhe_sig[9];

  static const int ringmx = 5;
  static const int sectmx = 12;
  static const int routmx = 36;
  static const int rout12mx = 24;
  static const int neffip = 6;

  TProfile* com_hotime[ringmx][sectmx];
  TProfile* com_hopedtime[ringmx][sectmx];

  TProfile* com_hbtime[ringmx][sectmx];

  TH1F* com_corrsglb[ringmx][sectmx];
  TH1F* com_corrsgrb[ringmx][sectmx];
  TH1F* com_corrsglu[ringmx][sectmx];
  TH1F* com_corrsgru[ringmx][sectmx];
  TH1F* com_corrsgall[ringmx][sectmx];

  TH1F* com_corrsgl[ringmx][sectmx];
  TH1F* com_corrsgr[ringmx][sectmx];

  TH1F* com_corrsgc[ringmx][sectmx];

  TH1F* com_sigrsg[ringmx][routmx + 1];
  TH1F* com_crossg[ringmx][routmx + 1];
  float com_invang[ringmx][routmx + 1];

  TH1F* ped_evt;
  TH1F* ped_mean;
  TH1F* ped_width;
  TH1F* fit_chi;
  TH1F* sig_evt;
  TH1F* fit_sigevt;
  TH1F* fit_bkgevt;
  TH1F* sig_mean;
  TH1F* sig_diff;
  TH1F* sig_width;
  TH1F* sig_sigma;
  TH1F* sig_meanerr;
  TH1F* sig_meanerrp;
  TH1F* sig_signf;

  TH1F* ped_statmean;
  TH1F* sig_statmean;
  TH1F* ped_rms;
  TH1F* sig_rms;

  TH2F* const_eta_phi;

  TH1F* const_eta[netamx];
  TH1F* stat_eta[netamx];
  TH1F* statmn_eta[netamx];
  TH1F* peak_eta[netamx];

  TH1F* const_hpdrm[ringmx];
  //  TH1F* stat_hpdrm[ringmx];
  //  TH1F* statmn_hpdrm[ringmx];
  TH1F* peak_hpdrm[ringmx];

  TH1F* mean_eta_ave;
  TH1F* mean_phi_ave;
  TH1F* mean_phi_hst;

  TH2F* sig_effi[neffip];
  TH2F* mean_energy;

  double fitprm[nsgpr][netamx];

  TProfile* sigvsevt[15][ncut];

  //  int   irun, ievt, itrg1, itrg2, isect, nrecht, nfound, nlost, ndof, nmuon;
  unsigned ievt, hoflag;
  int irun, ilumi, nprim, isect, isect2, ndof, nmuon;

  float inslumi, trkdr, trkdz, trkvx, trkvy, trkvz, trkmm, trkth, trkph, chisq, therr, pherr, hodx, hody, hoang, htime,
      hosig[9], hocorsig[18], hocro, hbhesig[9], caloen[3];
  float momatho, tkpt03, ecal03, hcal03;

  int Nevents;
  int nbn;
  float alow;
  float ahigh;
  float binwid;
  int irunold;

  edm::InputTag hoCalibVariableCollectionTag;
  edm::EDGetTokenT<HOCalibVariableCollection> tok_ho_;
  edm::EDGetTokenT<HORecHitCollection> tok_allho_;
  // ----------member data ---------------------------
};

const int HOCalibAnalyzer::ringmx;
const int HOCalibAnalyzer::sectmx;
const int HOCalibAnalyzer::routmx;
const int HOCalibAnalyzer::rout12mx;
const int HOCalibAnalyzer::neffip;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

HOCalibAnalyzer::HOCalibAnalyzer(const edm::ParameterSet& iConfig)
// It is very likely you want the following in your configuration
// hoCalibVariableCollectionTag = cms.InputTag('hoCalibProducer', 'HOCalibVariableCollection')
{
  tok_ho_ = consumes<HOCalibVariableCollection>(iConfig.getParameter<edm::InputTag>("hoCalibVariableCollectionTag"));
  tok_allho_ = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInputTag"));
  //now do what ever initialization is needed
  ipass = 0;
  Nevents = 0;

  theRootFileName = iConfig.getUntrackedParameter<std::string>("RootFileName", "test.root");
  theoutputtxtFile = iConfig.getUntrackedParameter<std::string>("txtFileName", "test.txt");
  theoutputpsFile = iConfig.getUntrackedParameter<std::string>("psFileName", "test.ps");

  m_allHOsignal = iConfig.getUntrackedParameter<bool>("allsignal", false);
  m_hbinfo = iConfig.getUntrackedParameter<bool>("hbinfo", false);
  m_hbtime = iConfig.getUntrackedParameter<bool>("hbtime", false);
  m_hotime = iConfig.getUntrackedParameter<bool>("hotime", false);
  m_correl = iConfig.getUntrackedParameter<bool>("correl", false);
  m_checkmap = iConfig.getUntrackedParameter<bool>("checkmap", false);
  m_combined = iConfig.getUntrackedParameter<bool>("combined", false);
  m_constant = iConfig.getUntrackedParameter<bool>("get_constant", false);
  m_figure = iConfig.getUntrackedParameter<bool>("get_figure", true);
  m_histfit = iConfig.getUntrackedParameter<bool>("histFit", true);
  m_pedsuppr = iConfig.getUntrackedParameter<bool>("pedSuppr", true);
  m_cosmic = iConfig.getUntrackedParameter<bool>("cosmic", true);
  m_sigma = iConfig.getUntrackedParameter<double>("sigma", 0.05);

  edm::Service<TFileService> fs;

  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();

  T1 = new TTree("T1", "DT+CSC+HO");

  T1->Branch("irun", &irun, "irun/I");
  T1->Branch("ievt", &ievt, "ievt/i");

  //  T1->Branch("itrg1",&itrg1,"itrg1/I");
  //  T1->Branch("itrg2",&itrg2,"itrg2/I");

  T1->Branch("isect", &isect, "isect/I");
  T1->Branch("isect2", &isect2, "isect2/I");
  T1->Branch("ndof", &ndof, "ndof/I");
  T1->Branch("nmuon", &nmuon, "nmuon/I");

  T1->Branch("ilumi", &ilumi, "ilumi/I");
  if (!m_cosmic) {
    T1->Branch("inslumi", &inslumi, "inslumi/F");
    T1->Branch("nprim", &nprim, "nprim/I");
    T1->Branch("tkpt03", &tkpt03, " tkpt03/F");
    T1->Branch("ecal03", &ecal03, " ecal03/F");
    T1->Branch("hcal03", &hcal03, " hcal03/F");
  }

  T1->Branch("trkdr", &trkdr, "trkdr/F");
  T1->Branch("trkdz", &trkdz, "trkdz/F");

  T1->Branch("trkvx", &trkvx, "trkvx/F");
  T1->Branch("trkvy", &trkvy, "trkvy/F");
  T1->Branch("trkvz", &trkvz, "trkvz/F");
  T1->Branch("trkmm", &trkmm, "trkmm/F");
  T1->Branch("trkth", &trkth, "trkth/F");
  T1->Branch("trkph", &trkph, "trkph/F");

  T1->Branch("chisq", &chisq, "chisq/F");
  T1->Branch("therr", &therr, "therr/F");
  T1->Branch("pherr", &pherr, "pherr/F");
  T1->Branch("hodx", &hodx, "hodx/F");
  T1->Branch("hody", &hody, "hody/F");
  T1->Branch("hoang", &hoang, "hoang/F");

  T1->Branch("momatho", &momatho, "momatho/F");
  T1->Branch("hoflag", &hoflag, "hoflag/i");
  T1->Branch("htime", &htime, "htime/F");
  T1->Branch("hosig", hosig, "hosig[9]/F");
  T1->Branch("hocro", &hocro, "hocro/F");
  T1->Branch("hocorsig", hocorsig, "hocorsig[18]/F");
  T1->Branch("caloen", caloen, "caloen[3]/F");

  if (m_hbinfo) {  // #ifdef HBINFO
    T1->Branch("hbhesig", hbhesig, "hbhesig[9]/F");
  }  //m_hbinfo #endif

  char name[200];
  char title[200];

  if (m_allHOsignal) {
    ho_entry = fs->make<TH2F>(
        "ho_entry", "ho entry", netamx + 1, -netamx / 2 - 0.5, netamx / 2 + 0.5, nphimx, 0.5, nphimx + 0.5);

    ho_energy = fs->make<TH2F>(
        "ho_energy", "ho energy (GeV)", netamx + 1, -netamx / 2 - 0.5, netamx / 2 + 0.5, nphimx, 0.5, nphimx + 0.5);

    ho_energy2 = fs->make<TH2F>("ho_energy2",
                                "ho energy2 (GeV*GeV)",
                                netamx + 1,
                                -netamx / 2 - 0.5,
                                netamx / 2 + 0.5,
                                nphimx,
                                0.5,
                                nphimx + 0.5);

    ho_rms = fs->make<TH2F>(
        "ho_rms", "ho rms (GeV)", netamx + 1, -netamx / 2 - 0.5, netamx / 2 + 0.5, nphimx, 0.5, nphimx + 0.5);

    for (int ij = 0; ij < netamx; ij++) {
      int ieta = getHOieta(ij);
      for (int jk = 0; jk < nphimx; jk++) {
        sprintf(name, "ho_indenergy_%i_%i", ij, jk);
        sprintf(title, "ho IndEnergy (GeV) i#eta=%i i#phi=%i", ieta, jk + 1);
        ho_indenergy[ij][jk] = fs->make<TH1F>(name, title, 120, -5., 55.);
      }
    }
  }

  muonnm = fs->make<TH1F>("muonnm", "No of muon", 10, -0.5, 9.5);
  muonmm = fs->make<TH1F>("muonmm", "P_{mu}", 200, -100., 100.);
  muonth = fs->make<TH1F>("muonth", "{Theta}_{mu}", 180, 0., 180.);
  muonph = fs->make<TH1F>("muonph", "{Phi}_{mu}", 180, -180., 180.);
  muonch = fs->make<TH1F>("muonch", "{chi^2}/ndf", 100, 0., 1000.);

  sel_muonnm = fs->make<TH1F>("sel_muonnm", "No of muon(sel)", 10, -0.5, 9.5);
  sel_muonmm = fs->make<TH1F>("sel_muonmm", "P_{mu}(sel)", 200, -100., 100.);
  sel_muonth = fs->make<TH1F>("sel_muonth", "{Theta}_{mu}(sel)", 180, 0., 180.);
  sel_muonph = fs->make<TH1F>("sel_muonph", "{Phi}_{mu}(sel)", 180, -180., 180.);
  sel_muonch = fs->make<TH1F>("sel_muonch", "{chi^2}/ndf(sel)", 100, 0., 1000.);

  int nbin = 50;  //40;// 45; //50; //55; //60; //55; //45; //40; //50;
  alow = -2.0;    // -1.85; //-1.90; // -1.95; // -2.0;
  ahigh = 8.0;    // 8.15; // 8.10; //  8.05; //  8.0;

  float tmpwid = (ahigh - alow) / nbin;
  nbn = int(-alow / tmpwid) + 1;
  if (nbn < 0)
    nbn = 0;
  if (nbn > nbin)
    nbn = nbin;

  edm::LogInfo("HOCalib") << "nbin " << nbin << " " << alow << " " << ahigh << " " << tmpwid << " " << nbn;

  for (int ij = 0; ij < 15; ij++) {
    sprintf(title, "sigvsndof_ring%i", ij + 1);
    sigvsevt[ij][0] = fs->make<TProfile>(title, title, 50, 0., 50., -9., 20.);

    sprintf(title, "sigvschisq_ring%i", ij + 1);
    sigvsevt[ij][1] = fs->make<TProfile>(title, title, 50, 0., 30., -9., 20.);

    sprintf(title, "sigvsth_ring%i", ij + 1);
    sigvsevt[ij][2] = fs->make<TProfile>(title, title, 50, .7, 2.4, -9., 20.);

    sprintf(title, "sigvsph_ring%i", ij + 1);
    sigvsevt[ij][3] = fs->make<TProfile>(title, title, 50, -2.4, -0.7, -9., 20.);

    sprintf(title, "sigvstherr_ring%i", ij + 1);
    sigvsevt[ij][4] = fs->make<TProfile>(title, title, 50, 0., 0.2, -9., 20.);

    sprintf(title, "sigvspherr_ring%i", ij + 1);
    sigvsevt[ij][5] = fs->make<TProfile>(title, title, 50, 0., 0.2, -9., 20.);

    sprintf(title, "sigvsdircos_ring%i", ij + 1);
    sigvsevt[ij][6] = fs->make<TProfile>(title, title, 50, 0.5, 1., -9., 20.);

    sprintf(title, "sigvstrkmm_ring%i", ij + 1);
    sigvsevt[ij][7] = fs->make<TProfile>(title, title, 50, 0., 50., -9., 20.);

    sprintf(title, "sigvsnmuon_ring%i", ij + 1);
    sigvsevt[ij][8] = fs->make<TProfile>(title, title, 5, 0.5, 5.5, -9., 20.);

    sprintf(title, "sigvserr_ring%i", ij + 1);
    sigvsevt[ij][9] = fs->make<TProfile>(title, title, 50, 0., .3, -9., 20.);

    sprintf(title, "sigvsaccx_ring%i", ij + 1);
    sigvsevt[ij][10] = fs->make<TProfile>(title, title, 100, -25., 25., -9., 20.);

    sprintf(title, "sigvsaccy_ring%i", ij + 1);
    sigvsevt[ij][11] = fs->make<TProfile>(title, title, 100, -25., 25., -9., 20.);

    sprintf(title, "sigvscalo_ring%i", ij + 1);
    sigvsevt[ij][12] = fs->make<TProfile>(title, title, 100, 0., 15., -9., 20.);
  }

  for (int jk = 0; jk < netamx; jk++) {
    int ieta = (jk < 15) ? jk + 1 : 14 - jk;
    for (int ij = 0; ij < nphimx + 1; ij++) {
      if (ij == nphimx) {
        sprintf(title, "sig_eta%i_allphi", ieta);
      } else {
        sprintf(title, "sig_eta%i_phi%i", ieta, ij + 1);
      }
      sigrsg[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);
      if (ij == nphimx) {
        sprintf(title, "ped_eta%i_allphi", ieta);
      } else {
        sprintf(title, "ped_eta%i_phi%i", ieta, ij + 1);
      }
      crossg[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);
    }

    for (int ij = 0; ij < nphimx; ij++) {
      if (m_hotime) {  //#ifdef HOTIME
        sprintf(title, "hotime_eta%i_phi%i", (jk <= 14) ? jk + 1 : 14 - jk, ij + 1);
        hotime[jk][ij] = fs->make<TProfile>(title, title, 10, -0.5, 9.5, -1.0, 30.0);

        sprintf(title, "hopedtime_eta%i_phi%i", (jk <= 14) ? jk + 1 : 14 - jk, ij + 1);
        hopedtime[jk][ij] = fs->make<TProfile>(title, title, 10, -0.5, 9.5, -1.0, 30.0);

      }                //m_hotime #endif
      if (m_hbtime) {  //#ifdef HBTIME
        sprintf(title, "hbtime_eta%i_phi%i", (jk <= 15) ? jk + 1 : 15 - jk, ij + 1);
        hbtime[jk][ij] = fs->make<TProfile>(title, title, 10, -0.5, 9.5, -1.0, 30.0);
      }  //m_hbtime #endif

      if (m_correl) {  //#ifdef CORREL
        sprintf(title, "corrsg_eta%i_phi%i_leftbottom", ieta, ij + 1);
        corrsglb[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

        sprintf(title, "corrsg_eta%i_phi%i_rightbottom", ieta, ij + 1);
        corrsgrb[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

        sprintf(title, "corrsg_eta%i_phi%i_leftup", ieta, ij + 1);
        corrsglu[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

        sprintf(title, "corrsg_eta%i_phi%i_rightup", ieta, ij + 1);
        corrsgru[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

        sprintf(title, "corrsg_eta%i_phi%i_all", ieta, ij + 1);
        corrsgall[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

        sprintf(title, "corrsg_eta%i_phi%i_left", ieta, ij + 1);
        corrsgl[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

        sprintf(title, "corrsg_eta%i_phi%i_right", ieta, ij + 1);
        corrsgr[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);
      }                  //m_correl #endif
      if (m_checkmap) {  // #ifdef CHECKMAP
        sprintf(title, "corrsg_eta%i_phi%i_centrl", ieta, ij + 1);
        corrsgc[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);
      }  //m_checkmap #endif
    }
  }

  mnsigrsg = fs->make<TH1F>(
      "mnsigrsg", "mnsigrsg", netamx * nphimx + ringmx * routmx, -0.5, netamx * nphimx + ringmx * routmx - 0.5);
  rmssigrsg = fs->make<TH1F>(
      "rmssigrsg", "rmssigrsg", netamx * nphimx + ringmx * routmx, -0.5, netamx * nphimx + ringmx * routmx - 0.5);
  nevsigrsg = fs->make<TH1F>(
      "nevsigrsg", "nevsigrsg", netamx * nphimx + ringmx * routmx, -0.5, netamx * nphimx + ringmx * routmx - 0.5);

  mncrossg = fs->make<TH1F>(
      "mncrossg", "mncrossg", netamx * nphimx + ringmx * routmx, -0.5, netamx * nphimx + ringmx * routmx - 0.5);
  rmscrossg = fs->make<TH1F>(
      "rmscrossg", "rmscrossg", netamx * nphimx + ringmx * routmx, -0.5, netamx * nphimx + ringmx * routmx - 0.5);
  nevcrossg = fs->make<TH1F>(
      "nevcrossg", "nevcrossg", netamx * nphimx + ringmx * routmx, -0.5, netamx * nphimx + ringmx * routmx - 0.5);

  for (int ij = 0; ij < neffip; ij++) {
    if (ij == 0) {
      sprintf(title, "Total projected muon in tower");
      sprintf(name, "total_evt");
    } else {
      sprintf(title, "Efficiency with sig >%i #sigma", ij);
      sprintf(name, "Effi_with_gt%i_sig", ij);
    }
    sig_effi[ij] =
        fs->make<TH2F>(name, title, netamx + 1, -netamx / 2 - 0.5, netamx / 2 + 0.5, nphimx, 0.5, nphimx + 0.5);
  }

  sprintf(title, "Mean Energy of all towers");
  sprintf(name, "mean_energy");
  mean_energy = fs->make<TH2F>(name, title, netamx + 1, -netamx / 2 - 0.5, netamx / 2 + 0.5, nphimx, 0.5, nphimx + 0.5);

  if (m_correl) {  //#ifdef CORREL
    mncorrsglb = fs->make<TH1F>("mncorrsglb", "mncorrsglb", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsglb = fs->make<TH1F>("rmscorrsglb", "rmscorrsglb", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsglb = fs->make<TH1F>("nevcorrsglb", "nevcorrsglb", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);

    mncorrsgrb = fs->make<TH1F>("mncorrsgrb", "mncorrsgrb", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsgrb = fs->make<TH1F>("rmscorrsgrb", "rmscorrsgrb", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsgrb = fs->make<TH1F>("nevcorrsgrb", "nevcorrsgrb", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);

    mncorrsglu = fs->make<TH1F>("mncorrsglu", "mncorrsglu", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsglu = fs->make<TH1F>("rmscorrsglu", "rmscorrsglu", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsglu = fs->make<TH1F>("nevcorrsglu", "nevcorrsglu", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);

    mncorrsgru = fs->make<TH1F>("mncorrsgru", "mncorrsgru", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsgru = fs->make<TH1F>("rmscorrsgru", "rmscorrsgru", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsgru = fs->make<TH1F>("nevcorrsgru", "nevcorrsgru", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);

    mncorrsgall = fs->make<TH1F>("mncorrsgall", "mncorrsgall", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsgall = fs->make<TH1F>("rmscorrsgall", "rmscorrsgall", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsgall = fs->make<TH1F>("nevcorrsgall", "nevcorrsgall", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);

    mncorrsgl = fs->make<TH1F>("mncorrsgl", "mncorrsgl", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsgl = fs->make<TH1F>("rmscorrsgl", "rmscorrsgl", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsgl = fs->make<TH1F>("nevcorrsgl", "nevcorrsgl", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);

    mncorrsgr = fs->make<TH1F>("mncorrsgr", "mncorrsgr", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsgr = fs->make<TH1F>("rmscorrsgr", "rmscorrsgr", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsgr = fs->make<TH1F>("nevcorrsgr", "nevcorrsgr", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
  }  //m_correl #endif

  if (m_checkmap) {  //#ifdef CHECKMAP
    mncorrsgc = fs->make<TH1F>("mncorrsgc", "mncorrsgc", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    rmscorrsgc = fs->make<TH1F>("rmscorrsgc", "rmscorrsgc", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
    nevcorrsgc = fs->make<TH1F>("nevcorrsgc", "nevcorrsgc", netamx * nphimx + 60, -0.5, netamx * nphimx + 59.5);
  }  //m_checkmap #endif

  if (m_combined) {  //#ifdef COMBINED
    for (int jk = 0; jk < ringmx; jk++) {
      for (int ij = 0; ij < routmx + 1; ij++) {
        if (jk != 2 && ij > rout12mx)
          continue;
        int phmn = 3 * ij - 1;
        int phmx = 3 * ij + 1;
        if (jk == 2) {
          phmn = 2 * ij - 1;
          phmx = 2 * ij;
        }
        if (phmn <= 0)
          phmn = nphimx + phmn;
        if (phmx <= 0)
          phmx = nphimx + phmx;

        if ((jk == 2 && ij == routmx) || (jk != 2 && ij == rout12mx)) {
          sprintf(title, "sig_ring%i_allrm", jk - 2);
          sprintf(name, "sig_ring%i_allrm", jk - 2);
        } else {
          sprintf(title, "sig_ring%i_phi%i-%i", jk - 2, phmn, phmx);
          sprintf(name, "sig_ring%i_rout%i", jk - 2, ij + 1);
        }
        com_sigrsg[jk][ij] = fs->make<TH1F>(name, title, nbin, alow, ahigh);
        if ((jk == 2 && ij == routmx) || (jk != 2 && ij == rout12mx)) {
          sprintf(title, "ped_ring%i_allrm", jk - 2);
          sprintf(name, "ped_ring%i_allrm", jk - 2);
        } else {
          sprintf(title, "ped_ring%i_phi%i-%i", jk - 2, phmn, phmx);
          sprintf(name, "ped_ring%i_rout%i", jk - 2, ij + 1);
        }
        com_crossg[jk][ij] = fs->make<TH1F>(name, title, nbin, alow, ahigh);
      }

      for (int ij = 0; ij < sectmx; ij++) {
        if (m_hotime) {  //#ifdef HOTIME
          sprintf(title, "com_hotime_ring%i_sect%i", jk - 2, ij + 1);
          com_hotime[jk][ij] = fs->make<TProfile>(title, title, 10, -0.5, 9.5, -1.0, 30.0);

          sprintf(title, "com_hopedtime_ring%i_sect%i", jk - 2, ij + 1);
          com_hopedtime[jk][ij] = fs->make<TProfile>(title, title, 10, -0.5, 9.5, -1.0, 30.0);
        }                //m_hotime #endif
        if (m_hbtime) {  //#ifdef HBTIME
          sprintf(title, "_com_hbtime_ring%i_serrct%i", jk - 2, ij + 1);
          com_hbtime[jk][ij] = fs->make<TProfile>(title, title, 10, -0.5, 9.5, -1.0, 30.0);
        }  //m_hbtime #endif

        if (m_correl) {  //#ifdef CORREL
          sprintf(title, "com_corrsg_ring%i_sect%i_leftbottom", jk - 2, ij + 1);
          com_corrsglb[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

          sprintf(title, "com_corrsg_ring%i_sect%i_rightbottom", jk - 2, ij + 1);
          com_corrsgrb[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

          sprintf(title, "com_corrsg_ring%i_sect%i_leftup", jk - 2, ij + 1);
          com_corrsglu[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

          sprintf(title, "com_corrsg_ring%i_sect%i_rightup", jk - 2, ij + 1);
          com_corrsgru[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

          sprintf(title, "com_corrsg_ring%i_sect%i_all", jk - 2, ij + 1);
          com_corrsgall[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

          sprintf(title, "com_corrsg_ring%i_sect%i_left", jk - 2, ij + 1);
          com_corrsgl[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

          sprintf(title, "com_corrsg_ring%i_sect%i_right", jk - 2, ij + 1);
          com_corrsgr[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);
        }  //m_correl #endif

        if (m_checkmap) {  // #ifdef CHECKMAP
          sprintf(title, "com_corrsg_ring%i_sect%i_centrl", jk - 2, ij + 1);
          com_corrsgc[jk][ij] = fs->make<TH1F>(title, title, nbin, alow, ahigh);
        }  //m_checkmap #endif
      }
    }
  }  //m_combined #endif

  for (int ij = -1; ij <= 1; ij++) {
    for (int jk = -1; jk <= 1; jk++) {
      int kl = 3 * (ij + 1) + jk + 1;

      sprintf(title, "hosct2p_eta%i_phi%i", ij, jk);
      ho_sig2p[kl] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

      sprintf(title, "hosct1p_eta%i_phi%i", ij, jk);
      ho_sig1p[kl] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

      sprintf(title, "hosct00_eta%i_phi%i", ij, jk);
      ho_sig00[kl] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

      sprintf(title, "hosct1m_eta%i_phi%i", ij, jk);
      ho_sig1m[kl] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

      sprintf(title, "hosct2m_eta%i_phi%i", ij, jk);
      ho_sig2m[kl] = fs->make<TH1F>(title, title, nbin, alow, ahigh);

      if (m_hbinfo) {  // #ifdef HBINFO
        sprintf(title, "hbhesig_eta%i_phi%i", ij, jk);
        hbhe_sig[kl] = fs->make<TH1F>(title, title, 51, -10.5, 40.5);
      }  //m_hbinfo #endif
    }
  }

  if (m_constant) {
    ped_evt = fs->make<TH1F>("ped_evt", "ped_evt", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    ped_mean = fs->make<TH1F>("ped_mean", "ped_mean", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    ped_width = fs->make<TH1F>("ped_width", "ped_width", netamx * nphimx, -0.5, netamx * nphimx - 0.5);

    fit_chi = fs->make<TH1F>("fit_chi", "fit_chi", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_evt = fs->make<TH1F>("sig_evt", "sig_evt", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    fit_sigevt = fs->make<TH1F>("fit_sigevt", "fit_sigevt", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    fit_bkgevt = fs->make<TH1F>("fit_bkgevt", "fit_bkgevt", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_mean = fs->make<TH1F>("sig_mean", "sig_mean", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_diff = fs->make<TH1F>("sig_diff", "sig_diff", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_width = fs->make<TH1F>("sig_width", "sig_width", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_sigma = fs->make<TH1F>("sig_sigma", "sig_sigma", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_meanerr = fs->make<TH1F>("sig_meanerr", "sig_meanerr", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_meanerrp = fs->make<TH1F>("sig_meanerrp", "sig_meanerrp", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_signf = fs->make<TH1F>("sig_signf", "sig_signf", netamx * nphimx, -0.5, netamx * nphimx - 0.5);

    ped_statmean = fs->make<TH1F>("ped_statmean", "ped_statmean", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_statmean = fs->make<TH1F>("sig_statmean", "sig_statmean", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    ped_rms = fs->make<TH1F>("ped_rms", "ped_rms", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
    sig_rms = fs->make<TH1F>("sig_rms", "sig_rms", netamx * nphimx, -0.5, netamx * nphimx - 0.5);

    const_eta_phi = fs->make<TH2F>(
        "const_eta_phi", "const_eta_phi", netamx + 1, -(netamx + 1) / 2., (netamx + 1) / 2., nphimx, 0.5, nphimx + 0.5);

    for (int ij = 0; ij < netamx; ij++) {
      int ieta = (ij < 15) ? ij + 1 : 14 - ij;
      sprintf(title, "Cont_Eta_%i", ieta);
      const_eta[ij] = fs->make<TH1F>(title, title, nphimx, 0.5, nphimx + 0.5);

      sprintf(title, "Peak_Eta_%i", ieta);
      peak_eta[ij] = fs->make<TH1F>(title, title, nphimx, 0.5, nphimx + 0.5);
    }

    for (int ij = 0; ij < ringmx; ij++) {
      int iring = ij - 2;
      int iread = (ij == 2) ? routmx : rout12mx;
      sprintf(title, "Cont_hpdrm_%i", iring);
      const_hpdrm[ij] = fs->make<TH1F>(title, title, iread, 0.5, iread + 0.5);

      sprintf(title, "Peak_hpdrm_%i", iring);
      peak_hpdrm[ij] = fs->make<TH1F>(title, title, iread, 0.5, iread + 0.5);
    }

    mean_phi_hst = fs->make<TH1F>("mean_phi_hst", "mean_phi_hst", netamx + 1, -(netamx + 1) / 2., (netamx + 1) / 2.);
    mean_phi_ave = fs->make<TH1F>("mean_phi_ave", "mean_phi_ave", netamx + 1, -(netamx + 1) / 2., (netamx + 1) / 2.);

    mean_eta_ave = fs->make<TH1F>("mean_eta_ave", "mean_eta_ave", nphimx, 0.5, nphimx + 0.5);

  }  //m_constant

  for (int ij = 0; ij < netamx; ij++) {
    int ieta = (ij < 15) ? ij + 1 : 14 - ij;

    sprintf(title, "Stat_Eta_%i", ieta);
    stat_eta[ij] = fs->make<TH1F>(title, title, nphimx, 0.5, nphimx + 0.5);

    sprintf(title, "#mu(stat)_Eta_%i", ieta);
    statmn_eta[ij] = fs->make<TH1F>(title, title, nphimx, 0.5, nphimx + 0.5);
  }

  for (int jk = 0; jk < netamx; jk++) {
    for (int ij = 0; ij < nphimx; ij++) {
      invang[jk][ij] = 0.0;
    }
  }
  for (int jk = 0; jk < ringmx; jk++) {
    for (int ij = 0; ij < routmx + 1; ij++) {
      com_invang[jk][ij] = 0.0;
    }
  }
}

HOCalibAnalyzer::~HOCalibAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

  theFile->cd();
  theFile->Write();
  theFile->Close();
  edm::LogInfo("HOCalib") << " Ttoal events = " << Nevents << " Selected events # is " << ipass;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void HOCalibAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // calcualte these once (and avoid the pow(int,int) ambiguities for c++)
  int mypow_2_0 = 1;  // 2^0
  int mypow_2_1 = 2;  // 2^1
  int mypow_2_2 = 4;  // 2^2

  int mypow_2_3 = 8;      // 2^3
  int mypow_2_4 = 16;     // 2^4
  int mypow_2_5 = 32;     // 2^5
  int mypow_2_6 = 64;     // 2^6
  int mypow_2_7 = 128;    // 2^7
  int mypow_2_8 = 256;    // 2^8
  int mypow_2_9 = 512;    // 2^9
  int mypow_2_10 = 1024;  // 2^10
  int mypow_2_11 = 2048;  // 2^11
  int mypow_2_12 = 4096;  // 2^12

  /*
  //FIXGM Put this is initialiser
  int mapx1[6][3]={{1,4,8}, {12,7,3}, {5,9,13}, {11,6,2}, {16,15,14}, {19,18,17}}; 
  //    int etamap1[21]={-1, 0,3,1, 0,2,3, 1,0,2, -1, 3,1,2, 4,4,4, 5,5,5, -1};
  //  int phimap1[21]={-1, 0,2,2, 1,0,1, 1,2,1, -1, 0,0,2, 2,1,0, 2,1,0,-1};

    int mapx2[6][3]={{1,4,8}, {12,7,3}, {5,9,13}, {11,6,2}, {16,15,14}, {-1,-1,-1}};
  //  int etamap2[21]={-1, 0,3,1, 0,2,3, 1,0,2, -1, 3,1,2, 4,4,4, -1,-1,-1, -1};
  //  int phimap2[21]={-1, 0,2,2, 1,0,1, 1,2,1, -1, 0,0,2, 2,1,0,  2, 1, 0, -1};

    int mapx0p[9][2]={{3,1}, {7,4}, {6,5},  {12,8}, {0,0}, {11,9}, {16,13}, {15,14}, {19,17}};
    int mapx0m[9][2]={{17,19}, {14,15}, {13,16}, {9,11}, {0,0}, {8,12}, {5,6}, {4,7}, {1,3}};

  //  int etamap0p[21]={-1, 0,-1,0, 1,2,2, 1,3,5, -1, 5,3,6, 7,7,6, 8,-1,8, -1};
  //  int phimap0p[21]={-1, 1,-1,0, 1,1,0, 0,1,1, -1, 0,0,1, 1,0,0, 1,-1,0, -1};

  //  int etamap0m[21]={-1, 8,-1,8, 7,6,6, 7,5,3, -1, 3,5,2, 1,1,2, 0,-1,0, -1};
  //  int phimap0m[21]={-1, 0,-1,1, 0,0,1, 1,0,0, -1, 1,1,0, 0,1,1, 0,-1,1, -1};

    int etamap[4][21]={{-1, 0,3,1, 0,2,3, 1,0,2, -1, 3,1,2, 4,4,4, -1,-1,-1, -1}, //etamap2
		       {-1, 0,3,1, 0,2,3, 1,0,2, -1, 3,1,2, 4,4,4, 5,5,5, -1},    //etamap1 
		       {-1, 0,-1,0, 1,2,2, 1,3,5, -1, 5,3,6, 7,7,6, 8,-1,8, -1},  //etamap0p
		       {-1, 8,-1,8, 7,6,6, 7,5,3, -1, 3,5,2, 1,1,2, 0,-1,0, -1}}; //etamap0m

    int phimap[4][21] ={{-1, 0,2,2, 1,0,1, 1,2,1, -1, 0,0,2, 2,1,0, 2,1,0, -1},    //phimap2
			{-1, 0,2,2, 1,0,1, 1,2,1, -1, 0,0,2, 2,1,0, 2,1,0, -1},    //phimap1
			{-1, 1,-1,0, 1,1,0, 0,1,1, -1, 0,0,1, 1,0,0, 1,-1,0, -1},  //phimap0p
			{-1, 0,-1,1, 0,0,1, 1,0,0, -1, 1,1,0, 0,1,1, 0,-1,1, -1}};  //phimap0m
  //swapped phi map for R0+/R0- (15/03/07)  
  for (int ij=0; ij<4; ij++) {
    for (int jk=0; jk<21; jk++) {
      edm::LogInfo("HOCalib") <<"ieta "<<ij<<" "<<jk<<" "<<etamap[ij][jk];
    }
  }

  // Character convention for R+/-1/2
  //      int npixleft[21] = {-1, F, Q,-1, M, D, J,-1, T,-1, C,-1, R, P, H,-1, N, G};
  //      int npixrigh[21] = { Q, S,-1, D, J, L,-1, K,-1, E,-1, P, H, B,-1, G, A,-1};
  
  //      int npixlb1[21]={-1,-1,-1,-1, F, Q, S,-1, M, J, L, T, K,-1, C, R, P, H};
  //      int npixrb1[21]={-1,-1,-1, F, Q, S,-1, M, D, L,-1, K,-1, C, E, P, H, B};
  //      int npixlu1[21]={ M, D, J, T, K,-1, C,-1, R, H, B,-1, N, G, A,-1,-1,-1};
  //      int npixru1[21]={ D, J, L, K,-1, C, E, R, P, B,-1, N, G, A,-1,-1,-1,-1};

  int npixleft[21]={0, 0, 1, 2, 0, 4, 5, 6, 0, 8, 0, 0,11, 0,13,14,15, 0,17,18,0};
  int npixrigh[21]={0, 2, 3, 0, 5, 6, 7, 0, 9, 0, 0,12, 0,14,15,16, 0,18,19, 0,0};
  int npixlebt[21]={0, 0, 0, 0, 0, 1, 2, 3, 0, 4, 0, 6, 7, 8, 9, 0,11,13,14,15,0};
  int npixribt[21]={0, 0, 0, 0, 1, 2, 3, 0, 4, 5, 0, 7, 0, 9, 0,11,12,14,15,16,0};
  int npixleup[21]={0, 4, 5, 6, 8, 9, 0,11, 0,13, 0,15,16, 0,17,18,19, 0, 0, 0,0};
  int npixriup[21]={0, 5, 6, 7, 9, 0,11,12,13,14, 0,16, 0,17,18,19, 0, 0, 0, 0,0};
  */

  int iaxxx = 0;
  int ibxxx = 0;

  Nevents++;

  using namespace edm;

  float pival = acos(-1.);
  irunold = irun = iEvent.id().run();
  ievt = iEvent.id().event();
  ilumi = iEvent.luminosityBlock();

  if (m_allHOsignal) {
    edm::Handle<HORecHitCollection> hoht;
    iEvent.getByToken(tok_allho_, hoht);
    if (hoht.isValid() && !(*hoht).empty()) {
      ho_entry->Fill(-1., -1.);  //Count of total number of entries
      for (HORecHitCollection::const_iterator ij = (*hoht).begin(); ij != (*hoht).end(); ij++) {
        HcalDetId id = (*ij).id();
        int tmpeta = id.ieta();
        int tmpphi = id.iphi();
        float signal = (*ij).energy();
        ho_entry->Fill(tmpeta, tmpphi);
        ho_energy->Fill(tmpeta, tmpphi, signal);
        ho_energy2->Fill(tmpeta, tmpphi, signal * signal);

        int inveta = invert_HOieta(tmpeta);
        ho_indenergy[inveta][tmpphi - 1]->Fill(signal);
      }
    }
  }

  edm::Handle<HOCalibVariableCollection> HOCalib;
  bool isCosMu = true;
  try {
    iEvent.getByToken(tok_ho_, HOCalib);

  } catch (cms::Exception& iEvent) {
    isCosMu = false;
  }
  if (Nevents % 5000 == 1)
    edm::LogInfo("HOCalib") << "nmuon event # " << Nevents << " Run # " << iEvent.id().run() << " Evt # "
                            << iEvent.id().event() << " " << ipass;

  if (isCosMu && !(*HOCalib).empty()) {
    nmuon = (*HOCalib).size();
    for (HOCalibVariableCollection::const_iterator hoC = (*HOCalib).begin(); hoC != (*HOCalib).end(); hoC++) {
      //      itrg1 = (*hoC).trig1;
      //      itrg2 = (*hoC).trig2;
      trkdr = (*hoC).trkdr;
      trkdz = (*hoC).trkdz;

      trkvx = (*hoC).trkvx;
      trkvy = (*hoC).trkvy;
      trkvz = (*hoC).trkvz;

      trkmm = (*hoC).trkmm;
      trkth = (*hoC).trkth;
      trkph = (*hoC).trkph;

      ndof = (int)(*hoC).ndof;
      //      nrecht = (int)(*hoC).nrecht;
      chisq = (*hoC).chisq;
      momatho = (*hoC).momatho;

      therr = (*hoC).therr;
      pherr = (*hoC).pherr;
      trkph = (*hoC).trkph;

      if (!m_cosmic) {
        nprim = (*hoC).nprim;
        inslumi = (*hoC).inslumi;
        tkpt03 = (*hoC).tkpt03;
        ecal03 = (*hoC).ecal03;
        hcal03 = (*hoC).hcal03;
      }

      isect = (*hoC).isect;
      isect2 = (*hoC).isect2;
      hodx = (*hoC).hodx;
      hody = (*hoC).hody;
      hoang = (*hoC).hoang;
      htime = (*hoC).htime;
      hoflag = (*hoC).hoflag;
      for (int ij = 0; ij < 9; ij++) {
        hosig[ij] = (*hoC).hosig[ij];
      }  //edm::LogInfo("HOCalib")<<"hosig "<<i<<" "<<hosig[ij];}
      for (int ij = 0; ij < 18; ij++) {
        hocorsig[ij] = (*hoC).hocorsig[ij];
      }  // edm::LogInfo("HOCalib")<<"hocorsig "<<i<<" "<<hocorsig[ij];}
      hocro = (*hoC).hocro;
      for (int ij = 0; ij < 3; ij++) {
        caloen[ij] = (*hoC).caloen[ij];
      }

      if (m_hbinfo) {
        for (int ij = 0; ij < 9; ij++) {
          hbhesig[ij] = (*hoC).hbhesig[ij];
        }
      }  // edm::LogInfo("HOCalib")<<"hbhesig "<<ij<<" "<<hbhesig[ij];}}

      T1->Fill();

      int ipsall = 0;
      int ips0 = 0;
      int ips1 = 0;
      int ips2 = 0;
      int ips3 = 0;
      int ips4 = 0;
      int ips5 = 0;
      int ips6 = 0;
      int ips7 = 0;
      int ips8 = 0;
      int ips9 = 0;
      int ips10 = 0;
      int ips11 = 0;
      int ips12 = 0;

      //      int iselect3 = 0;
      //      if (ndof >=15 && chisq <30) iselect3 = 1;

      if (isect < 0)
        continue;  //FIXGM Is it proper place ?
      if (fabs(trkth - pival / 2) < 0.000001)
        continue;  //22OCT07

      int ieta = int((abs(isect) % 10000) / 100.) - 50;  //an offset to acodate -ve eta values
      if (abs(ieta) >= 16)
        continue;
      int iphi = abs(isect) % 100;

      int tmpsect = int((iphi + 1) / 6.) + 1;
      if (tmpsect > 12)
        tmpsect = 1;

      int iring = 0;
      int tmpeta = ieta + 4;  //For pixel mapping
      if (ieta >= -15 && ieta <= -11) {
        iring = -2;
        tmpeta = -11 - ieta;
      }  //abs(ieta)-11;}
      if (ieta >= -10 && ieta <= -5) {
        iring = -1;
        tmpeta = -5 - ieta;
      }  // abs(ieta)-5;}
      if (ieta >= 5 && ieta <= 10) {
        iring = 1;
        tmpeta = ieta - 5;
      }
      if (ieta >= 11 && ieta <= 15) {
        iring = 2;
        tmpeta = ieta - 11;
      }

      int iring2 = iring + 2;

      int tmprout = (iring == 0) ? int((iphi + 1) / 2.) + 1 : int((iphi + 1) / 3.) + 1;
      int tmproutmx = (iring == 0) ? routmx : rout12mx;
      if (tmprout > tmproutmx)
        tmprout = 1;

      // CRUZET1
      if (m_cosmic) {
        /*  GMA temoparily change to increase event size at 3 & 9 O'clock position */
        if (abs(ndof) >= 20 && abs(ndof) < 40) {
          ips0 = (int)mypow_2_0;
          ipsall += ips0;
        }
        if (chisq > 0 && chisq < 15) {
          ips1 = (int)mypow_2_1;
          ipsall += ips1;
        }  //18Jan2008
        if (fabs(trkth - pival / 2) < 21.5) {
          ips2 = (int)mypow_2_2;
          ipsall += ips2;
        }  //No nead for pp evt
        if (fabs(trkph + pival / 2) < 21.5) {
          ips3 = (int)mypow_2_3;
          ipsall += ips3;
        }  //No nead for pp evt

        if (therr < 0.02) {
          ips4 = (int)mypow_2_4;
          ipsall += ips4;
        }
        if (pherr < 0.0002) {
          ips5 = (int)mypow_2_5;
          ipsall += ips5;
        }
        if (fabs(hoang) > 0.30) {
          ips6 = (int)mypow_2_6;
          ipsall += ips6;
        }
        if (fabs(trkmm) > 0.100) {
          ips7 = (int)mypow_2_7;
          ipsall += ips7;
        }
        //      if (nmuon ==1)               {ips8 = (int)mypow_2_8;  ipsall +=ips8;}
        if (nmuon >= 1 && nmuon <= 4) {
          ips8 = (int)mypow_2_8;
          ipsall += ips8;
        }

        if (iring2 == 2) {
          if (fabs(hodx) < 100 && fabs(hodx) > 2 && fabs(hocorsig[8]) < 40 && fabs(hocorsig[8]) > 2) {
            ips10 = (int)mypow_2_10;
            ipsall += ips10;
          }

          if (fabs(hody) < 100 && fabs(hody) > 2 && fabs(hocorsig[9]) < 40 && fabs(hocorsig[9]) > 2) {
            ips11 = (int)mypow_2_11;
            ipsall += ips11;
          }

        } else {
          if (fabs(hodx) < 100 && fabs(hodx) > 2) {
            ips10 = (int)mypow_2_10;
            ipsall += ips10;
          }

          if (fabs(hody) < 100 && fabs(hody) > 2) {
            ips11 = (int)mypow_2_11;
            ipsall += ips11;
          }
        }
        if (caloen[0] == 0) {
          ips12 = (int)mypow_2_12;
          ipsall += ips12;
        }
      } else {
        //csa08
        if (abs(ndof) >= 20 && abs(ndof) < 40) {
          ips0 = (int)mypow_2_0;
          ipsall += ips0;
        }
        if (chisq > 0 && chisq < 15) {
          ips1 = (int)mypow_2_1;
          ipsall += ips1;
        }  //18Jan2008
        if (fabs(trkth - pival / 2) < 21.5) {
          ips2 = (int)mypow_2_2;
          ipsall += ips2;
        }  //No nead for pp evt
        if (fabs(trkph + pival / 2) < 21.5) {
          ips3 = (int)mypow_2_3;
          ipsall += ips3;
        }  //No nead for pp evt

        if (therr < 0.02) {
          ips4 = (int)mypow_2_4;
          ipsall += ips4;
        }
        if (pherr < 0.0002) {
          ips5 = (int)mypow_2_5;
          ipsall += ips5;
        }
        if (fabs(hoang) > 0.30) {
          ips6 = (int)mypow_2_6;
          ipsall += ips6;
        }
        if (fabs(trkmm) > 4.0) {
          ips7 = (int)mypow_2_7;
          ipsall += ips7;
        }
        if (nmuon >= 1 && nmuon <= 2) {
          ips8 = (int)mypow_2_8;
          ipsall += ips8;
        }

        if (iring2 == 2) {
          if (fabs(hodx) < 100 && fabs(hodx) > 2 && fabs(hocorsig[8]) < 40 && fabs(hocorsig[8]) > 2) {
            ips10 = (int)mypow_2_10;
            ipsall += ips10;
          }

          if (fabs(hody) < 100 && fabs(hody) > 2 && fabs(hocorsig[9]) < 40 && fabs(hocorsig[9]) > 2) {
            ips11 = (int)mypow_2_11;
            ipsall += ips11;
          }

        } else {
          if (fabs(hodx) < 100 && fabs(hodx) > 2) {
            ips10 = (int)mypow_2_10;
            ipsall += ips10;
          }

          if (fabs(hody) < 100 && fabs(hody) > 2) {
            ips11 = (int)mypow_2_11;
            ipsall += ips11;
          }
        }
        //	if (m_cosmic || (caloen[0] >0.5 && caloen[0]<5.0)) {ips12=(int)pow_2_12;ipsall +=ips12;}
        if (ndof > 0 && caloen[0] < 5.0) {
          ips12 = (int)mypow_2_12;
          ipsall += ips12;
        }
        /*      */
      }

      if (htime > -40 && htime < 60) {
        ips9 = (int)mypow_2_9;
        ipsall += ips9;
      }

      if (ipsall - ips0 == mypow_2_ncut - mypow_2_0 - 1)
        sigvsevt[iring2][0]->Fill(abs(ndof), hosig[4]);
      if (ipsall - ips1 == mypow_2_ncut - mypow_2_1 - 1)
        sigvsevt[iring2][1]->Fill(chisq, hosig[4]);
      if (ipsall - ips2 == mypow_2_ncut - mypow_2_2 - 1)
        sigvsevt[iring2][2]->Fill(trkth, hosig[4]);
      if (ipsall - ips3 == mypow_2_ncut - mypow_2_3 - 1)
        sigvsevt[iring2][3]->Fill(trkph, hosig[4]);
      if (ipsall - ips4 == mypow_2_ncut - mypow_2_4 - 1)
        sigvsevt[iring2][4]->Fill(therr, hosig[4]);
      if (ipsall - ips5 == mypow_2_ncut - mypow_2_5 - 1)
        sigvsevt[iring2][5]->Fill(pherr, hosig[4]);
      if (ipsall - ips6 == mypow_2_ncut - mypow_2_6 - 1)
        sigvsevt[iring2][6]->Fill(hoang, hosig[4]);
      if (ipsall - ips7 == mypow_2_ncut - mypow_2_7 - 1)
        sigvsevt[iring2][7]->Fill(fabs(trkmm), hosig[4]);
      if (ipsall - ips8 == mypow_2_ncut - mypow_2_8 - 1)
        sigvsevt[iring2][8]->Fill(nmuon, hosig[4]);
      if (ipsall - ips9 == mypow_2_ncut - mypow_2_9 - 1)
        sigvsevt[iring2][9]->Fill(htime, hosig[4]);
      if (ipsall - ips10 == mypow_2_ncut - mypow_2_10 - 1)
        sigvsevt[iring2][10]->Fill(hodx, hosig[4]);
      if (ipsall - ips11 == mypow_2_ncut - mypow_2_11 - 1)
        sigvsevt[iring2][11]->Fill(hody, hosig[4]);
      if (!m_cosmic) {
        if (ipsall - ips12 == mypow_2_ncut - mypow_2_12 - 1)
          sigvsevt[iring2][12]->Fill(caloen[0], hosig[4]);
      }

      sigvsevt[iring2 + 5][0]->Fill(abs(ndof), hosig[4]);
      if (ips0 > 0) {
        sigvsevt[iring2 + 5][1]->Fill(chisq, hosig[4]);
        if (ips1 > 0) {
          sigvsevt[iring2 + 5][2]->Fill(trkth, hosig[4]);
          if (ips2 > 0) {
            sigvsevt[iring2 + 5][3]->Fill(trkph, hosig[4]);
            if (ips3 > 0) {
              sigvsevt[iring2 + 5][4]->Fill(therr, hosig[4]);
              if (ips4 > 0) {
                sigvsevt[iring2 + 5][5]->Fill(pherr, hosig[4]);
                if (ips5 > 0) {
                  sigvsevt[iring2 + 5][6]->Fill(hoang, hosig[4]);
                  if (ips6 > 0) {
                    sigvsevt[iring2 + 5][7]->Fill(fabs(trkmm), hosig[4]);
                    if (ips7 > 0) {
                      sigvsevt[iring2 + 5][8]->Fill(nmuon, hosig[4]);
                      if (ips8 > 0) {
                        sigvsevt[iring2 + 5][9]->Fill(htime, hosig[4]);
                        if (ips9 > 0) {
                          sigvsevt[iring2 + 5][10]->Fill(hodx, hosig[4]);
                          if (ips10 > 0) {
                            sigvsevt[iring2 + 5][11]->Fill(hody, hosig[4]);
                            if (ips11 > 0) {
                              if (!m_cosmic)
                                sigvsevt[iring2 + 5][12]->Fill(caloen[0], hosig[4]);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      sigvsevt[iring2 + 10][0]->Fill(abs(ndof), hosig[4]);
      sigvsevt[iring2 + 10][1]->Fill(chisq, hosig[4]);
      sigvsevt[iring2 + 10][2]->Fill(trkth, hosig[4]);
      sigvsevt[iring2 + 10][3]->Fill(trkph, hosig[4]);
      sigvsevt[iring2 + 10][4]->Fill(therr, hosig[4]);
      sigvsevt[iring2 + 10][5]->Fill(pherr, hosig[4]);
      sigvsevt[iring2 + 10][6]->Fill(hoang, hosig[4]);
      sigvsevt[iring2 + 10][7]->Fill(fabs(trkmm), hosig[4]);
      sigvsevt[iring2 + 10][8]->Fill(nmuon, hosig[4]);
      sigvsevt[iring2 + 10][9]->Fill(htime, hosig[4]);
      sigvsevt[iring2 + 10][10]->Fill(hodx, hosig[4]);
      sigvsevt[iring2 + 10][11]->Fill(hody, hosig[4]);
      if (!m_cosmic)
        sigvsevt[iring2 + 10][12]->Fill(caloen[0], hosig[4]);

      int iselect = (ipsall == mypow_2_ncut - 1) ? 1 : 0;

      if (hocro != -100.0 && hocro < -50.0)
        hocro += 100.;

      muonnm->Fill(nmuon);
      muonmm->Fill(trkmm);
      muonth->Fill(trkth * 180 / pival);
      muonph->Fill(trkph * 180 / pival);
      muonch->Fill(chisq);

      if (iselect == 1) {
        ipass++;
        sel_muonnm->Fill(nmuon);
        sel_muonmm->Fill(trkmm);
        sel_muonth->Fill(trkth * 180 / pival);
        sel_muonph->Fill(trkph * 180 / pival);
        sel_muonch->Fill(chisq);
      }

      //      if (iselect3) T1->Fill();

      int tmpphi = (iphi + 1) % 3;  //pixel mapping
      int npixel = 0;
      int itag = -1;
      int iflip = 0;
      int fact = 2;

      if (iring == 0) {
        tmpphi = (iphi + 1) % 2;
        if (tmpsect == 2 || tmpsect == 3 || tmpsect == 6 || tmpsect == 7 || tmpsect == 10 || tmpsect == 11) {
          npixel = mapx0p[tmpeta][tmpphi];
          itag = 2;
        } else {
          npixel = mapx0m[tmpeta][tmpphi];
          itag = 3;
        }
      } else {
        fact = 3;
        if (tmpsect % 2 == 1)
          iflip = 1;
        if (abs(iring) == 1) {
          npixel = mapx1[tmpeta][(iflip == 0) ? tmpphi : abs(tmpphi - 2)];
          itag = 1;
        } else {
          npixel = mapx2[tmpeta][(iflip == 0) ? tmpphi : abs(tmpphi - 2)];
          itag = 0;
        }
      }

      int tmpeta1 = (ieta > 0) ? ieta - 1 : -ieta + 14;

      int iselect2 = 0;
      if (hosig[4] != -100) {
        if (m_cosmic) {
          if (caloen[2] <= 0.0)
            iselect2 = 1;
        } else {
          if (caloen[2] <= 3.0)
            iselect2 = 1;
        }
      }

      //      edm::LogInfo("HOCalib") <<"cosmic "<<hosig[4]<<" "<<caloen[3]<<" "<<int(iselect2)<<" "<<int(m_cosmic);

      if (iselect2 == 1) {
        int tmpphi2 = (iphi + 6 <= nphimx) ? iphi + 5 : iphi + 5 - nphimx;

        int tmprout2 = (iring == 0) ? int((tmpphi2 + 2) / 2.) + 1 : int((tmpphi2 + 2) / 3.) + 1;
        if (tmprout2 > tmproutmx)
          tmprout2 = 1;

        if (cro_ssg[tmpeta1][tmpphi2].size() < 4000) {
          if (hocro > alow && hocro < ahigh) {
            if (!m_histfit)
              cro_ssg[tmpeta1][tmpphi2].push_back(hocro);
            crossg[tmpeta1][tmpphi2]->Fill(hocro);
          }
        }

        if (tmpphi2 >= 0 && tmpphi2 < nphimx) {
          crossg[tmpeta1][nphimx]->Fill(hocro);
        }
        if (m_combined) {
          com_crossg[iring2][tmprout2 - 1]->Fill(hocro);
          com_crossg[iring2][tmproutmx]->Fill(hocro);
        }
      }

      if (iselect == 1) {
        for (int ij = 0; ij < neffip; ij++) {
          if (ij == 0) {
            sig_effi[ij]->Fill(ieta, iphi, 1.);
          } else {
            if (hosig[4] > ij * m_sigma) {
              sig_effi[ij]->Fill(ieta, iphi, 1.);
            }
          }
        }

        //Histogram filling for noise study: phi shift according to DTChamberAnalysis
        int tmpphi1 = iphi - 1;

        if (sig_reg[tmpeta1][tmpphi1].size() < 4000) {
          if (hosig[4] > -50 && hosig[4] < 15) {
            sigrsg[tmpeta1][tmpphi1]->Fill(hosig[4]);
            if (!m_histfit && hosig[4] <= ahigh / 2.)
              sig_reg[tmpeta1][tmpphi1].push_back(hosig[4]);
            invang[tmpeta1][tmpphi1] += 1. / fabs(hoang);
          }
        }

        if (tmpphi1 >= 0 && tmpphi1 < nphimx) {  //GREN
          sigrsg[tmpeta1][nphimx]->Fill(hosig[4]);
          invang[tmpeta1][nphimx] += 1. / fabs(hoang);
        }

        if (m_combined) {  //#ifdef COMBINED
          com_sigrsg[iring2][tmprout - 1]->Fill(hosig[4]);
          com_invang[iring2][tmprout - 1] += 1. / fabs(hoang);

          com_sigrsg[iring2][tmproutmx]->Fill(hosig[4]);
          com_invang[iring2][tmproutmx] += 1. / fabs(hoang);
        }  //m_combined #endif

        if (m_checkmap || m_correl) {  //#ifdef CHECKMAP
          tmpeta = etamap[itag][npixel];
          tmpphi = phimap[itag][npixel];
          if (tmpeta >= 0 && tmpphi >= 0) {
            if (iflip != 0)
              tmpphi = abs(tmpphi - 2);
            if (int((hocorsig[fact * tmpeta + tmpphi] - hosig[4]) * 10000) / 10000. != 0) {
              iaxxx++;
              edm::LogInfo("HOCalib") << "iring2xxx " << irun << " " << ievt << " " << isect << " " << iring << " "
                                      << tmpsect << " " << ieta << " " << iphi << " " << npixel << " " << tmpeta << " "
                                      << tmpphi << " " << tmpeta1 << " " << tmpphi1 << " itag " << itag << " " << iflip
                                      << " " << fact << " " << hocorsig[fact * tmpeta + tmpphi] << " "
                                      << fact * tmpeta + tmpphi << " " << hosig[4] << " " << hodx << " " << hody;

              for (int ij = 0; ij < 18; ij++) {
                edm::LogInfo("HOCalib") << " " << ij << " " << hocorsig[ij];
              }
              edm::LogInfo("HOCalib") << " ix " << iaxxx << " " << ibxxx;
            } else {
              ibxxx++;
            }

            corrsgc[tmpeta1][tmpphi1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            if (m_combined) {  //#ifdef COMBINED
              com_corrsgc[iring2][tmpsect - 1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            }  //m_combined #endif
          }
        }  //m_checkmap #endif

        if (m_correl) {  //#ifdef CORREL
          float allcorsig = 0.0;

          tmpeta = etamap[itag][npixleft[npixel]];
          tmpphi = phimap[itag][npixleft[npixel]];

          if (tmpeta >= 0 && tmpphi >= 0) {
            if (iflip != 0)
              tmpphi = abs(tmpphi - 2);
            corrsgl[tmpeta1][tmpphi1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            allcorsig += hocorsig[fact * tmpeta + tmpphi];
            if (m_combined) {  //#ifdef COMBINED
              com_corrsgl[iring2][tmpsect - 1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            }  //m_combined #endif
          }

          tmpeta = etamap[itag][npixrigh[npixel]];
          tmpphi = phimap[itag][npixrigh[npixel]];
          if (tmpeta >= 0 && tmpphi >= 0) {
            if (iflip != 0)
              tmpphi = abs(tmpphi - 2);
            corrsgr[tmpeta1][tmpphi1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            allcorsig += hocorsig[fact * tmpeta + tmpphi];
            if (m_combined) {  // #ifdef COMBINED
              com_corrsgr[iring2][tmpsect - 1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            }  //m_combined #endif
          }

          tmpeta = etamap[itag][npixlebt[npixel]];
          tmpphi = phimap[itag][npixlebt[npixel]];
          if (tmpeta >= 0 && tmpphi >= 0) {
            if (iflip != 0)
              tmpphi = abs(tmpphi - 2);
            corrsglb[tmpeta1][tmpphi1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            allcorsig += hocorsig[fact * tmpeta + tmpphi];
            if (m_combined) {  //#ifdef COMBINED
              com_corrsglb[iring2][tmpsect - 1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            }  //m_combined #endif
          }

          tmpeta = etamap[itag][npixribt[npixel]];
          tmpphi = phimap[itag][npixribt[npixel]];
          if (tmpeta >= 0 && tmpphi >= 0) {
            if (iflip != 0)
              tmpphi = abs(tmpphi - 2);
            corrsgrb[tmpeta1][tmpphi1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            allcorsig += hocorsig[fact * tmpeta + tmpphi];
            if (m_combined) {  // #ifdef COMBINED
              com_corrsgrb[iring2][tmpsect - 1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            }  //m_combined #endif
          }

          tmpeta = etamap[itag][npixleup[npixel]];
          tmpphi = phimap[itag][npixleup[npixel]];
          if (tmpeta >= 0 && tmpphi >= 0) {
            if (iflip != 0)
              tmpphi = abs(tmpphi - 2);
            corrsglu[tmpeta1][tmpphi1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            allcorsig += hocorsig[fact * tmpeta + tmpphi];
            if (m_combined) {  // #ifdef COMBINED
              com_corrsglu[iring2][tmpsect - 1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            }  //m_combined #endif
          }

          tmpeta = etamap[itag][npixriup[npixel]];
          tmpphi = phimap[itag][npixriup[npixel]];
          if (tmpeta >= 0 && tmpphi >= 0) {
            if (iflip != 0)
              tmpphi = abs(tmpphi - 2);
            corrsgru[tmpeta1][tmpphi1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            allcorsig += hocorsig[fact * tmpeta + tmpphi];
            if (m_combined) {  // #ifdef COMBINED
              com_corrsgru[iring2][tmpsect - 1]->Fill(hocorsig[fact * tmpeta + tmpphi]);
            }  //m_combined #endif
          }
          corrsgall[tmpeta1][tmpphi1]->Fill(allcorsig);
          if (m_combined) {  // #ifdef COMBINED
            com_corrsgall[iring2][tmpsect - 1]->Fill(allcorsig);
          }  //m_combined #endif

        }  //m_correl #endif
        for (int k = 0; k < 9; k++) {
          switch (iring) {
            case 2:
              ho_sig2p[k]->Fill(hosig[k]);
              break;
            case 1:
              ho_sig1p[k]->Fill(hosig[k]);
              break;
            case 0:
              ho_sig00[k]->Fill(hosig[k]);
              break;
            case -1:
              ho_sig1m[k]->Fill(hosig[k]);
              break;
            case -2:
              ho_sig2m[k]->Fill(hosig[k]);
              break;
          }
          if (m_hbinfo) {  // #ifdef HBINFO
            hbhe_sig[k]->Fill(hbhesig[k]);
            //	    edm::LogInfo("HOCalib") <<"hbhe "<<k<<" "<<hbhesig[k];
          }  //m_hbinfo #endif
        }
      }  //if (iselect==1)

    }  //for (HOCalibVariableCollection::const_iterator hoC=(*HOCalib).begin(); hoC!=(*HOCalib).end(); hoC++){

  }  //if (isCosMu)
}

// ------------ method called once each job just before starting event loop  ------------
void HOCalibAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void HOCalibAnalyzer::endJob() {
  theFile->cd();

  if (m_allHOsignal) {
    for (int jk = 0; jk < ho_energy->GetNbinsX(); jk++) {
      for (int kl = 0; kl < ho_energy->GetNbinsY(); kl++) {
        double entry = ho_entry->GetBinContent(jk + 1, kl + 1);
        if (entry < 1.)
          entry = 1.;

        double energy = ho_energy->GetBinContent(jk + 1, kl + 1) / entry;
        double energy2 = ho_energy2->GetBinContent(jk + 1, kl + 1) / entry;
        double rms = sqrt(energy2 - energy * energy);

        double xval = ho_energy->GetXaxis()->GetBinCenter(jk + 1);
        double yval = ho_energy->GetYaxis()->GetBinCenter(kl + 1);

        ho_rms->Fill(xval, yval, rms);
      }
    }
  }

  for (int ij = 0; ij < nphimx; ij++) {
    for (int jk = 0; jk < netamx; jk++) {
      nevsigrsg->Fill(netamx * ij + jk, sigrsg[jk][ij]->GetEntries());
      mnsigrsg->Fill(netamx * ij + jk, sigrsg[jk][ij]->GetMean());
      rmssigrsg->Fill(netamx * ij + jk, sigrsg[jk][ij]->GetRMS());

      nevcrossg->Fill(netamx * ij + jk, crossg[jk][ij]->GetEntries());
      mncrossg->Fill(netamx * ij + jk, crossg[jk][ij]->GetMean());
      rmscrossg->Fill(netamx * ij + jk, crossg[jk][ij]->GetRMS());

      if (m_correl) {  //#ifdef CORREL

        nevcorrsglb->Fill(netamx * ij + jk, corrsglb[jk][ij]->GetEntries());
        mncorrsglb->Fill(netamx * ij + jk, corrsglb[jk][ij]->GetMean());
        rmscorrsglb->Fill(netamx * ij + jk, corrsglb[jk][ij]->GetRMS());

        nevcorrsgrb->Fill(netamx * ij + jk, corrsgrb[jk][ij]->GetEntries());
        mncorrsgrb->Fill(netamx * ij + jk, corrsgrb[jk][ij]->GetMean());
        rmscorrsgrb->Fill(netamx * ij + jk, corrsgrb[jk][ij]->GetRMS());

        nevcorrsglu->Fill(netamx * ij + jk, corrsglu[jk][ij]->GetEntries());
        mncorrsglu->Fill(netamx * ij + jk, corrsglu[jk][ij]->GetMean());
        rmscorrsglu->Fill(netamx * ij + jk, corrsglu[jk][ij]->GetRMS());

        nevcorrsgru->Fill(netamx * ij + jk, corrsgru[jk][ij]->GetEntries());
        mncorrsgru->Fill(netamx * ij + jk, corrsgru[jk][ij]->GetMean());
        rmscorrsgru->Fill(netamx * ij + jk, corrsgru[jk][ij]->GetRMS());

        nevcorrsgall->Fill(netamx * ij + jk, corrsgall[jk][ij]->GetEntries());
        mncorrsgall->Fill(netamx * ij + jk, corrsgall[jk][ij]->GetMean());
        rmscorrsgall->Fill(netamx * ij + jk, corrsgall[jk][ij]->GetRMS());

        nevcorrsgl->Fill(netamx * ij + jk, corrsgl[jk][ij]->GetEntries());
        mncorrsgl->Fill(netamx * ij + jk, corrsgl[jk][ij]->GetMean());
        rmscorrsgl->Fill(netamx * ij + jk, corrsgl[jk][ij]->GetRMS());

        nevcorrsgr->Fill(netamx * ij + jk, corrsgr[jk][ij]->GetEntries());
        mncorrsgr->Fill(netamx * ij + jk, corrsgr[jk][ij]->GetMean());
        rmscorrsgr->Fill(netamx * ij + jk, corrsgr[jk][ij]->GetRMS());
      }                  //m_correl #endif
      if (m_checkmap) {  //#ifdef CHECKMAP
        nevcorrsgc->Fill(netamx * ij + jk, corrsgc[jk][ij]->GetEntries());
        mncorrsgc->Fill(netamx * ij + jk, corrsgc[jk][ij]->GetMean());
        rmscorrsgc->Fill(netamx * ij + jk, corrsgc[jk][ij]->GetRMS());
      }  //m_checkmap #endif
    }
  }

  if (m_combined) {  // #ifdef COMBINED
    for (int jk = 0; jk < ringmx; jk++) {
      for (int ij = 0; ij < routmx; ij++) {
        if (jk != 2 && ij >= rout12mx)
          continue;
        nevsigrsg->Fill(netamx * nphimx + ringmx * ij + jk, com_sigrsg[jk][ij]->GetEntries());
        mnsigrsg->Fill(netamx * nphimx + ringmx * ij + jk, com_sigrsg[jk][ij]->GetMean());
        rmssigrsg->Fill(netamx * nphimx + ringmx * ij + jk, com_sigrsg[jk][ij]->GetRMS());

        nevcrossg->Fill(netamx * nphimx + ringmx * ij + jk, com_crossg[jk][ij]->GetEntries());
        mncrossg->Fill(netamx * nphimx + ringmx * ij + jk, com_crossg[jk][ij]->GetMean());
        rmscrossg->Fill(netamx * nphimx + ringmx * ij + jk, com_crossg[jk][ij]->GetRMS());
      }
    }

    for (int ij = 0; ij < sectmx; ij++) {
      for (int jk = 0; jk < ringmx; jk++) {
        if (m_correl) {  // #ifdef CORREL
          nevcorrsglb->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsglb[jk][ij]->GetEntries());
          mncorrsglb->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsglb[jk][ij]->GetMean());
          rmscorrsglb->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsglb[jk][ij]->GetRMS());

          nevcorrsgrb->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgrb[jk][ij]->GetEntries());
          mncorrsgrb->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgrb[jk][ij]->GetMean());
          rmscorrsgrb->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgrb[jk][ij]->GetRMS());

          nevcorrsglu->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsglu[jk][ij]->GetEntries());
          mncorrsglu->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsglu[jk][ij]->GetMean());
          rmscorrsglu->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsglu[jk][ij]->GetRMS());

          nevcorrsgru->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgru[jk][ij]->GetEntries());
          mncorrsgru->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgru[jk][ij]->GetMean());
          rmscorrsgru->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgru[jk][ij]->GetRMS());

          nevcorrsgall->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgall[jk][ij]->GetEntries());
          mncorrsgall->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgall[jk][ij]->GetMean());
          rmscorrsgall->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgall[jk][ij]->GetRMS());

          nevcorrsgl->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgl[jk][ij]->GetEntries());
          mncorrsgl->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgl[jk][ij]->GetMean());
          rmscorrsgl->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgl[jk][ij]->GetRMS());

          nevcorrsgr->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgr[jk][ij]->GetEntries());
          mncorrsgr->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgr[jk][ij]->GetMean());
          rmscorrsgr->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgr[jk][ij]->GetRMS());
        }                  //m_correl #endif
        if (m_checkmap) {  // #ifdef CHECKMAP
          nevcorrsgc->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgc[jk][ij]->GetEntries());
          mncorrsgc->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgc[jk][ij]->GetMean());
          rmscorrsgc->Fill(netamx * nphimx + ringmx * ij + jk, com_corrsgc[jk][ij]->GetRMS());
        }  //m_checkmap #endif
      }
    }
  }  //m_combined #endif

  for (int ij = 1; ij < neffip; ij++) {
    sig_effi[ij]->Divide(sig_effi[0]);
  }
  for (int ij = 0; ij < netamx; ij++) {
    for (int jk = 0; jk < nphimx; jk++) {
      int ieta = (ij < 15) ? ij + 1 : 14 - ij;
      int iphi = jk + 1;
      double signal = sigrsg[ij][jk]->GetMean();
      mean_energy->Fill(ieta, iphi, signal);
    }
  }

  int irunold = irun;

  gStyle->SetOptLogy(0);
  gStyle->SetTitleFillColor(10);
  gStyle->SetStatColor(10);

  gStyle->SetCanvasColor(10);
  gStyle->SetOptStat(0);  //1110);
  gStyle->SetOptTitle(1);

  gStyle->SetTitleColor(10);
  gStyle->SetTitleFontSize(0.09);
  gStyle->SetTitleOffset(-0.05);
  gStyle->SetTitleBorderSize(1);

  gStyle->SetPadColor(10);
  gStyle->SetPadBorderMode(0);
  gStyle->SetStatColor(10);
  gStyle->SetPadBorderMode(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatFontSize(.07);

  gStyle->SetStatStyle(1001);
  gStyle->SetOptFit(101);
  gStyle->SetCanvasColor(10);
  gStyle->SetCanvasBorderMode(0);

  gStyle->SetStatX(.99);
  gStyle->SetStatY(.99);
  gStyle->SetStatW(.45);
  gStyle->SetStatH(.16);
  gStyle->SetLabelSize(0.075, "XY");
  gStyle->SetLabelOffset(0.21, "XYZ");
  gStyle->SetTitleSize(0.065, "XY");
  gStyle->SetTitleOffset(0.06, "XYZ");
  gStyle->SetPadTopMargin(.09);
  gStyle->SetPadBottomMargin(0.11);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.15);
  gStyle->SetPadGridX(true);
  gStyle->SetPadGridY(true);
  gStyle->SetGridStyle(2);
  gStyle->SetNdivisions(303, "XY");

  gStyle->SetMarkerSize(0.60);
  gStyle->SetMarkerColor(2);
  gStyle->SetMarkerStyle(20);
  gStyle->SetTitleFontSize(0.07);

  char out_file[200];
  int xsiz = 700;
  int ysiz = 500;

  //   TCanvas *c2 = new TCanvas("c2", "Statistics and efficiency", xsiz, ysiz);
  //   c2->Divide(2,1); //(3,2);
  //   for (int ij=0; ij<neffip; ij=ij+3) {
  //     sig_effi[ij]->GetXaxis()->SetTitle("#eta");
  //     sig_effi[ij]->GetXaxis()->SetTitleSize(0.075);
  //     sig_effi[ij]->GetXaxis()->SetTitleOffset(0.65); //0.85
  //     sig_effi[ij]->GetXaxis()->CenterTitle();
  //     sig_effi[ij]->GetXaxis()->SetLabelSize(0.055);
  //     sig_effi[ij]->GetXaxis()->SetLabelOffset(0.001);

  //     sig_effi[ij]->GetYaxis()->SetTitle("#phi");
  //     sig_effi[ij]->GetYaxis()->SetTitleSize(0.075);
  //     sig_effi[ij]->GetYaxis()->SetTitleOffset(0.9);
  //     sig_effi[ij]->GetYaxis()->CenterTitle();
  //     sig_effi[ij]->GetYaxis()->SetLabelSize(0.055);
  //     sig_effi[ij]->GetYaxis()->SetLabelOffset(0.01);

  //     c2->cd(int(ij/3.)+1); sig_effi[ij]->Draw("colz");
  //   }
  //   sprintf(out_file, "comb_hosig_evt_%i.jpg",irunold);
  //   c2->SaveAs(out_file);

  //   gStyle->SetTitleFontSize(0.045);
  //   gStyle->SetPadRightMargin(0.1);
  //   gStyle->SetPadLeftMargin(0.1);
  //   gStyle->SetPadBottomMargin(0.12);

  //   TCanvas *c1 = new TCanvas("c1", "Mean signal in each tower", xsiz, ysiz);

  //   mean_energy->GetXaxis()->SetTitle("#eta");
  //   mean_energy->GetXaxis()->SetTitleSize(0.075);
  //   mean_energy->GetXaxis()->SetTitleOffset(0.65); //0.6
  //   mean_energy->GetXaxis()->CenterTitle();
  //   mean_energy->GetXaxis()->SetLabelSize(0.045);
  //   mean_energy->GetXaxis()->SetLabelOffset(0.001);

  //   mean_energy->GetYaxis()->SetTitle("#phi");
  //   mean_energy->GetYaxis()->SetTitleSize(0.075);
  //   mean_energy->GetYaxis()->SetTitleOffset(0.5);
  //   mean_energy->GetYaxis()->CenterTitle();
  //   mean_energy->GetYaxis()->SetLabelSize(0.045);
  //   mean_energy->GetYaxis()->SetLabelOffset(0.01);

  //   mean_energy->Draw("colz");
  //   sprintf(out_file, "homean_energy_%i.jpg",irunold);
  //   c1->SaveAs(out_file);

  //   delete c1;
  //   delete c2;

  gStyle->SetPadBottomMargin(0.14);
  gStyle->SetPadLeftMargin(0.17);
  gStyle->SetPadRightMargin(0.03);

  gStyle->SetOptStat(1110);

  const int nsample = 8;
  TF1* gx0[nsample] = {nullptr};
  TF1* ped0fun[nsample] = {nullptr};
  TF1* signal[nsample] = {nullptr};
  TF1* pedfun[nsample] = {nullptr};
  TF1* sigfun[nsample] = {nullptr};
  TF1* signalx[nsample] = {nullptr};

  TH1F* signall[nsample] = {nullptr};
  TH1F* pedstll[nsample] = {nullptr};

  if (m_constant) {
    gStyle->SetOptFit(101);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetStatBorderSize(1);
    gStyle->SetStatStyle(1001);
    gStyle->SetTitleColor(10);
    gStyle->SetTitleFontSize(0.09);
    gStyle->SetTitleOffset(-0.05);
    gStyle->SetTitleBorderSize(1);

    gStyle->SetCanvasColor(10);
    gStyle->SetPadColor(10);
    gStyle->SetStatColor(10);
    gStyle->SetStatFontSize(.07);
    gStyle->SetStatX(0.99);
    gStyle->SetStatY(0.99);
    gStyle->SetStatW(0.30);
    gStyle->SetStatH(0.10);
    gStyle->SetTitleSize(0.065, "XYZ");
    gStyle->SetLabelSize(0.075, "XYZ");
    gStyle->SetLabelOffset(0.012, "XYZ");
    gStyle->SetPadGridX(true);
    gStyle->SetPadGridY(true);
    gStyle->SetGridStyle(3);
    gStyle->SetNdivisions(101, "XY");
    gStyle->SetOptLogy(0);
    int iiter = 0;

    std::ofstream file_out(theoutputtxtFile.c_str());
    //    TPostScript* ps=0;
    int ips = 111;
    TPostScript ps(theoutputpsFile.c_str(), ips);
    ps.Range(20, 28);

    xsiz = 900;   //900;
    ysiz = 1200;  //600;
    TCanvas* c0 = new TCanvas("c0", " Pedestal vs signal", xsiz, ysiz);

    // Fix is done for eta-phi

    float mean_eta[netamx];
    float mean_phi[nphimx];
    float rms_eta[netamx];
    float rms_phi[nphimx];

    for (int ij = 0; ij < nphimx; ++ij) {
      mean_phi[ij] = 0;
      rms_phi[ij] = 0;
    }
    for (int ij = 0; ij < netamx; ++ij) {
      mean_eta[ij] = 0;
      rms_eta[ij] = 0;
    }

    int mxeta = 0;
    int mxphi = 0;
    int mneta = 0;
    int mnphi = 0;

    //iijj = 0 : Merging all ring
    //     = 1 : Individual HPD
    //iijj = 2 : merging all phi
    //     = 3 : Individual tower

    for (int iijj = 0; iijj < 4; iijj++) {
      //      if ((!mx_combined) && iijj==1) continue; //Use this only for combined data
      if (iijj == 0) {
        mxeta = ringmx;
        mxphi = 1;
        mneta = 0;
        mnphi = 0;
      } else if (iijj == 1) {
        mxeta = ringmx;
        mxphi = routmx;
        mneta = 0;
        mnphi = 0;
      } else if (iijj == 2) {
        mxeta = netamx;
        mxphi = 1;
        mneta = 0;
        mnphi = 0;
      } else if (iijj == 3) {
        mxeta = netamx;
        mxphi = nphimx;
        mneta = 0;
        mnphi = 0;
      }

      for (int jk = mneta; jk < mxeta; jk++) {
        for (int ij = mnphi; ij < mxphi; ij++) {
          if (iijj == 1)
            continue;
          if ((iijj == 0 || iijj == 1) && jk != 2 && ij >= rout12mx)
            continue;
          int izone = iiter % nsample;

          if (iijj == 0) {
            int iread = (jk == 2) ? routmx : rout12mx;
            signall[izone] = (TH1F*)com_sigrsg[jk][iread]->Clone("hnew");
            pedstll[izone] = (TH1F*)com_crossg[jk][iread]->Clone("hnew");
          } else if (iijj == 1) {
            signall[izone] = (TH1F*)com_sigrsg[jk][ij]->Clone("hnew");
            pedstll[izone] = (TH1F*)com_crossg[jk][ij]->Clone("hnew");
          } else if (iijj == 2) {
            signall[izone] = (TH1F*)sigrsg[jk][nphimx]->Clone("hnew");
            pedstll[izone] = (TH1F*)crossg[jk][nphimx]->Clone("hnew");
          } else if (iijj == 3) {
            signall[izone] = (TH1F*)sigrsg[jk][ij]->Clone("hnew");
            pedstll[izone] = (TH1F*)crossg[jk][ij]->Clone("hnew");
          }

          pedstll[izone]->SetLineWidth(2);
          signall[izone]->SetLineWidth(2);
          pedstll[izone]->SetLineColor(2);
          signall[izone]->SetLineColor(4);
          pedstll[izone]->SetNdivisions(506, "XY");
          signall[izone]->SetNdivisions(506, "XY");

          signall[izone]->GetXaxis()->SetLabelSize(.065);
          signall[izone]->GetYaxis()->SetLabelSize(.06);
          signall[izone]->GetXaxis()->SetTitle("Signal (GeV)");

          signall[izone]->GetXaxis()->SetTitleSize(.065);
          signall[izone]->GetXaxis()->CenterTitle();

          if (izone == 0) {  //iiter%8 ==0) {
            ps.NewPage();
            c0->Divide(4, 4);  //c0->Divide(2,4); // c0->Divide(1,2);
          }
          c0->cd(2 * izone + 1);  // (iiter%8)+1); //c0->cd(iiter%8+1);

          /*
	  if (iijj==0 && izone==0) {
	    gStyle->SetOptLogy(1);
	    gStyle->SetOptStat(0);
	    gStyle->SetOptFit(0);
	    c0->Divide(3,2);
	  }

	  if (iijj>0) {
	    gStyle->SetOptLogy(0);
	    gStyle->SetOptStat(1110);
	    gStyle->SetOptFit(101);
	    
	    if (iiter==0) {
	      int ips=111;
	      ps = new TPostScript(theoutputpsFile.c_str(),ips);
	      ps.Range(20,28);
	      xsiz = 900; //900;
	      ysiz = 1200; //600;
	      c0 = new TCanvas("c0", " Pedestal vs signal", xsiz, ysiz);
	    }
	    if (izone==0) {
	      ps.NewPage();
	      c0->Divide(4,4);
	    }
	  }
	  if (iijj==0) {c0->cd(izone+1); } else { c0->cd(2*izone+1);}
	  */

          float mean = pedstll[izone]->GetMean();
          float rms = pedstll[izone]->GetRMS();

          if (rms < 0.10)
            rms = 0.10;
          if (rms > 0.15)
            rms = 0.15;
          if (mean > 0.20)
            mean = 0.20;
          if (mean < -0.20)
            mean = -0.20;

          float xmn = mean - 6. * rms;
          float xmx = mean + 6. * rms;

          binwid = pedstll[izone]->GetBinWidth(1);
          if (xmx > pedstll[izone]->GetXaxis()->GetXmax())
            xmx = pedstll[izone]->GetXaxis()->GetXmax() - 0.5 * binwid;
          if (xmn < pedstll[izone]->GetXaxis()->GetXmin())
            xmn = pedstll[izone]->GetXaxis()->GetXmin() + 0.5 * binwid;

          float height = pedstll[izone]->GetEntries();

          double par[nbgpr] = {height, mean, 0.75 * rms};

          double gaupr[nbgpr];
          double parer[nbgpr];

          ietafit = jk;
          iphifit = ij;
          pedstll[izone]->GetXaxis()->SetLabelSize(.065);
          pedstll[izone]->GetYaxis()->SetLabelSize(.06);

          //	  if (iijj==0) {
          //	    pedstll[izone]->GetXaxis()->SetRangeUser(alow, ahigh);
          //	  } else {
          pedstll[izone]->GetXaxis()->SetRangeUser(xmn, xmx);
          //	  }

          if (iijj == 0) {
            pedstll[izone]->GetXaxis()->SetTitle("Pedestal/Signal (GeV)");
          } else {
            pedstll[izone]->GetXaxis()->SetTitle("Pedestal (GeV)");
          }
          pedstll[izone]->GetXaxis()->SetTitleSize(.065);
          pedstll[izone]->GetXaxis()->CenterTitle();
          //	  pedstll[izone]->SetLineWidth(2);

          pedstll[izone]->Draw();
          if (m_pedsuppr) {
            gaupr[0] = 0;
            gaupr[1] = 0.0;   // pedmean[ietafit][iphifit];
            gaupr[2] = 0.15;  //GMA need from database
            parer[0] = parer[1] = parer[2] = 0;
          } else {
            if (pedstll[izone]->GetEntries() > 5) {
              if ((iijj != 3) || m_histfit) {
                char temp[20];
                sprintf(temp, "gx0_%i", izone);
                gx0[izone] = new TF1(temp, gausX, xmn, xmx, nbgpr);
                gx0[izone]->SetParameters(par);
                gx0[izone]->SetLineWidth(1);
                pedstll[izone]->Fit(gx0[izone], "R+");

                for (int k = 0; k < nbgpr; k++) {
                  parer[k] = gx0[izone]->GetParError(k);
                  gaupr[k] = gx0[izone]->GetParameter(k);
                }
              } else {
                double strt[nbgpr] = {height, mean, 0.75 * rms};
                double step[nbgpr] = {1.0, 0.001, 0.001};
                double alowmn[nbgpr] = {0.5 * height, mean - rms, 0.3 * rms};
                double ahighmn[nbgpr] = {1.5 * height, mean + rms, 1.5 * rms};

                TMinuit* gMinuit = new TMinuit(nbgpr);
                gMinuit->SetFCN(fcnbg);

                double arglist[10];
                int ierflg = 0;
                arglist[0] = 0.5;
                gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
                char name[100];
                for (int k = 0; k < nbgpr; k++) {
                  sprintf(name, "pedpar%i", k);
                  gMinuit->mnparm(k, name, strt[k], step[k], alowmn[k], ahighmn[k], ierflg);
                }

                arglist[0] = 0;
                gMinuit->mnexcm("SIMPLEX", arglist, 0, ierflg);

                arglist[0] = 0;
                gMinuit->mnexcm("IMPROVE", arglist, 0, ierflg);

                TString chnam;
                double parv, err, xlo, xup, plerr, mierr, eparab, gcc;
                int iuit;

                for (int k = 0; k < nbgpr; k++) {
                  if (step[k] > -10) {
                    gMinuit->mnpout(k, chnam, parv, err, xlo, xup, iuit);
                    gMinuit->mnerrs(k, plerr, mierr, eparab, gcc);
                    //		    edm::LogInfo("HOCalib") <<"k "<< k<<" "<<chnam<<" "<<parv<<" "<<err<<" "<<xlo<<" "<<xup<<" "<<plerr<<" "<<mierr<<" "<<eparab;
                    if (k == 0) {
                      gaupr[k] = parv * binwid;
                      parer[k] = err * binwid;
                    } else {
                      gaupr[k] = parv;
                      parer[k] = err;
                    }
                  }
                }

                //		gx0[izone]->SetParameters(gaupr);

                char temp[20];
                sprintf(temp, "ped0fun_%i", izone);
                ped0fun[izone] = new TF1(temp, gausX, xmn, xmx, nbgpr);
                ped0fun[izone]->SetParameters(gaupr);
                ped0fun[izone]->SetLineColor(3);
                ped0fun[izone]->SetLineWidth(1);
                ped0fun[izone]->Draw("same");

                delete gMinuit;
              }
            } else {
              for (int k = 0; k < nbgpr; k++) {
                gaupr[k] = par[k];
              }
              gaupr[2] = 0.15;
            }
          }
          //	  if (iijj!=0)
          c0->cd(2 * izone + 2);
          if (signall[izone]->GetEntries() > 5) {
            Double_t parall[nsgpr];
            double parserr[nsgpr];
            double fitres[nsgpr];
            double pedht = 0;

            char temp[20];
            sprintf(temp, "signal_%i", izone);
            xmn = signall[izone]->GetXaxis()->GetXmin();
            xmx = 0.5 * signall[izone]->GetXaxis()->GetXmax();
            signal[izone] = new TF1(temp, totalfunc, xmn, xmx, nsgpr);
            xmx *= 2.0;
            if ((iijj != 3) || m_histfit) {
              pedht = (signall[izone]->GetBinContent(nbn - 1) + signall[izone]->GetBinContent(nbn) +
                       signall[izone]->GetBinContent(nbn + 1)) /
                      3.;

              if (m_pedsuppr) {
                parall[1] = 0.0;  // pedmean[ietafit][iphifit];
                parall[2] = 0.15;
              } else {
                for (int lm = 0; lm < nbgpr; lm++) {
                  parall[lm] = gaupr[lm];
                }
              }

              set_mean(parall[1], false);
              set_sigma(parall[2], false);

              parall[0] = 0.9 * pedht;  //GM for Z-mumu, there is almost no pedestal
              parall[3] = 0.14;
              double area = binwid * signall[izone]->GetEntries();
              parall[5] = area;

              if (iijj == 3) {
                parall[4] = fitprm[4][jk];
                parall[6] = fitprm[6][jk];
              } else {
                parall[4] = signall[izone]->GetMean();
                parall[6] = parall[2];
              }

              signal[izone]->SetParameters(parall);
              signal[izone]->FixParameter(1, parall[1]);
              signal[izone]->FixParameter(2, parall[2]);
              signal[izone]->SetParLimits(0, 0.00, 2.0 * pedht + 0.1);
              signal[izone]->FixParameter(3, 0.14);

              signal[izone]->SetParLimits(5, 0.40 * area, 1.15 * area);
              //	      if (m_histfit) { //GMA
              if (iijj == 3) {
                signal[izone]->SetParLimits(4, 0.2 * fitprm[4][jk], 2.0 * fitprm[4][jk]);
                signal[izone]->SetParLimits(6, 0.2 * fitprm[6][jk], 2.0 * fitprm[6][jk]);
              } else {
                signal[izone]->SetParLimits(4, 0.1, 1.0);
                signal[izone]->SetParLimits(6, 0.035, 0.3);
              }
              signal[izone]->SetParNames("const", "mean", "sigma", "Width", "MP", "Area", "GSigma");
              signall[izone]->Fit(signal[izone], "0R+");

              signall[izone]->GetXaxis()->SetRangeUser(xmn, xmx);
              for (int k = 0; k < nsgpr; k++) {
                fitres[k] = fitprm[k][jk] = signal[izone]->GetParameter(k);
                parserr[k] = signal[izone]->GetParError(k);
              }

            } else {
              double pedhtx = 0;
              for (unsigned i = 0; i < sig_reg[ietafit][iphifit].size(); i++) {
                if (sig_reg[ietafit][iphifit][ij] > gaupr[1] - 3 * gaupr[2] &&
                    sig_reg[ietafit][iphifit][ij] < gaupr[1] + gaupr[2])
                  pedhtx++;
              }

              set_mean(gaupr[1], false);
              set_sigma(gaupr[2], false);

              TString name[nsgpr] = {"const", "mean", "sigma", "Width", "MP", "Area", "GSigma"};
              double strt[nsgpr] = {0.9 * pedhtx,
                                    gaupr[1],
                                    gaupr[2],
                                    fitprm[3][jk],
                                    fitprm[4][jk],
                                    signall[izone]->GetEntries(),
                                    fitprm[6][jk]};
              double alowmn[nsgpr] = {
                  0.1 * pedhtx - 0.1, gaupr[1] - 0.1, gaupr[2] - 0.1, 0.07, 0.2 * strt[4], 0.1 * strt[5], 0.2 * strt[6]};
              double ahighmn[nsgpr] = {
                  1.2 * pedhtx + 0.1, gaupr[1] + 0.1, gaupr[2] + 0.1, 0.20, 2.5 * strt[4], 1.5 * strt[5], 2.2 * strt[6]};
              double step[nsgpr] = {1.0, 0.0, 0.0, 0.0, 0.001, 1.0, 0.002};

              TMinuit* gMinuit = new TMinuit(nsgpr);
              gMinuit->SetFCN(fcnsg);

              double arglist[10];
              int ierflg = 0;
              arglist[0] = 0.5;
              gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);

              for (int k = 0; k < nsgpr; k++) {
                gMinuit->mnparm(k, name[k], strt[k], step[k], alowmn[k], ahighmn[k], ierflg);
              }

              arglist[0] = 0;
              gMinuit->mnexcm("SIMPLEX", arglist, 0, ierflg);

              arglist[0] = 0;
              gMinuit->mnexcm("IMPROVE", arglist, 0, ierflg);

              TString chnam;
              double parv, err, xlo, xup, plerr, mierr, eparab, gcc;
              int iuit;

              for (int k = 0; k < nsgpr; k++) {
                if (step[k] > -10) {
                  gMinuit->mnpout(k, chnam, parv, err, xlo, xup, iuit);
                  gMinuit->mnerrs(k, plerr, mierr, eparab, gcc);
                  if (k == 0 || k == 5) {
                    fitres[k] = parv * binwid;
                    parserr[k] = err * binwid;
                  } else {
                    fitres[k] = parv;
                    parserr[k] = err;
                  }
                }
              }

              delete gMinuit;
            }

            //	    if (iijj==0) {
            //	      signall[izone]->Draw("same");
            //	    } else {
            signall[izone]->Draw();
            //	    }

            sprintf(temp, "pedfun_%i", izone);
            pedfun[izone] = new TF1(temp, gausX, xmn, xmx, nbgpr);
            pedfun[izone]->SetParameters(fitres);
            pedfun[izone]->SetLineColor(3);
            pedfun[izone]->SetLineWidth(1);
            pedfun[izone]->Draw("same");

            sprintf(temp, "signalfun_%i", izone);
            sigfun[izone] = new TF1(temp, langaufun, xmn, xmx, nsgpr - nbgpr);
            sigfun[izone]->SetParameters(&fitres[3]);
            sigfun[izone]->SetLineWidth(1);
            sigfun[izone]->SetLineColor(4);
            sigfun[izone]->Draw("same");

            sprintf(temp, "total_%i", izone);
            signalx[izone] = new TF1(temp, totalfunc, xmn, xmx, nsgpr);
            signalx[izone]->SetParameters(fitres);
            signalx[izone]->SetLineWidth(1);
            signalx[izone]->Draw("same");

            int kl = (jk < 15) ? jk + 1 : 14 - jk;

            edm::LogInfo("HOCalib") << "histinfo" << iijj << " fit " << std::setw(3) << kl << " " << std::setw(3)
                                    << ij + 1 << " " << std::setw(5) << pedstll[izone]->GetEntries() << " "
                                    << std::setw(6) << pedstll[izone]->GetMean() << " " << std::setw(6)
                                    << pedstll[izone]->GetRMS() << " " << std::setw(5) << signall[izone]->GetEntries()
                                    << " " << std::setw(6) << signall[izone]->GetMean() << " " << std::setw(6)
                                    << signall[izone]->GetRMS() << " " << std::setw(6) << signal[izone]->GetChisquare()
                                    << " " << std::setw(3) << signal[izone]->GetNDF();

            file_out << "histinfo" << iijj << " fit " << std::setw(3) << kl << " " << std::setw(3) << ij + 1 << " "
                     << std::setw(5) << pedstll[izone]->GetEntries() << " " << std::setw(6) << pedstll[izone]->GetMean()
                     << " " << std::setw(6) << pedstll[izone]->GetRMS() << " " << std::setw(5)
                     << signall[izone]->GetEntries() << " " << std::setw(6) << signall[izone]->GetMean() << " "
                     << std::setw(6) << signall[izone]->GetRMS() << " " << std::setw(6) << signal[izone]->GetChisquare()
                     << " " << std::setw(3) << signal[izone]->GetNDF() << std::endl;

            file_out << "fitres x" << iijj << " " << kl << " " << ij + 1 << " " << fitres[0] << " " << fitres[1] << " "
                     << fitres[2] << " " << fitres[3] << " " << fitres[4] << " " << fitres[5] << " " << fitres[6]
                     << std::endl;
            file_out << "parserr" << iijj << " " << kl << " " << ij + 1 << " " << parserr[0] << " " << parserr[1] << " "
                     << parserr[2] << " " << parserr[3] << " " << parserr[4] << " " << parserr[5] << " " << parserr[6]
                     << std::endl;

            int ieta = (jk < 15) ? (15 + jk) : (29 - jk);
            int ifl = nphimx * ieta + ij;

            if (iijj == 3) {
              ped_evt->Fill(ifl, pedstll[izone]->GetEntries());
              ped_mean->Fill(ifl, gaupr[1]);
              ped_width->Fill(ifl, gaupr[2]);
              fit_chi->Fill(ifl, signal[izone]->GetChisquare());
              sig_evt->Fill(ifl, signall[izone]->GetEntries());
              fit_sigevt->Fill(ifl, fitres[5]);
              fit_bkgevt->Fill(ifl, fitres[0] * sqrt(2 * acos(-1.)) * gaupr[2]);
              sig_mean->Fill(ifl, fitres[4]);
              sig_diff->Fill(ifl, fitres[4] - fitres[1]);
              sig_width->Fill(ifl, fitres[3]);
              sig_sigma->Fill(ifl, fitres[6]);
              sig_meanerr->Fill(ifl, parserr[4]);
              if (fitres[4] - fitres[1] != 0)
                sig_meanerrp->Fill(ifl, 100 * parserr[4] / (fitres[4] - fitres[1]));
              if (gaupr[2] != 0)
                sig_signf->Fill(ifl, (fitres[4] - fitres[1]) / gaupr[2]);

              ped_statmean->Fill(ifl, pedstll[izone]->GetMean());
              sig_statmean->Fill(ifl, signall[izone]->GetMean());
              ped_rms->Fill(ifl, pedstll[izone]->GetRMS());
              sig_rms->Fill(ifl, signall[izone]->GetRMS());
            }

            if ((iijj == 2) || (iijj == 3) || (iijj == 1)) {
              if (signall[izone]->GetEntries() > 5 && fitres[4] > 0.1) {
                //GMA need to put this==1 in future
                float fact = 0.812;
                if (abs(kl) <= 4)
                  fact = 0.895;
                fact *= 0.19;  //conversion factor for GeV/fC

                float fact2 = 0;
                if (iijj == 2)
                  fact2 = invang[jk][nphimx];
                if (iijj == 3)
                  fact2 = invang[jk][ij];
                if (iijj == 1)
                  fact2 = com_invang[jk][ij];

                float calibc = fact * fact2 / (fitres[4] * signall[izone]->GetEntries());
                float caliberr = TMath::Abs(calibc * parserr[4] / std::max(0.001, fitres[4]));

                if (iijj == 2) {
                  int ieta = (jk < 15) ? jk + 1 : 14 - jk;
                  mean_phi_hst->Fill(ieta, calibc);
                  mean_phi_hst->SetBinError(mean_phi_hst->FindBin(ieta), caliberr);
                  file_out << "intieta " << jk << " " << ij << " " << ieta << " " << mean_phi_hst->FindBin(double(ieta))
                           << " " << calibc << " " << caliberr << std::endl;
                } else if (iijj == 3) {
                  const_eta[jk]->Fill(ij + 1, calibc);
                  const_eta[jk]->SetBinError(const_eta[jk]->FindBin(ij + 1), caliberr);

                  peak_eta[jk]->Fill(ij + 1, fitres[4]);
                  peak_eta[jk]->SetBinError(peak_eta[jk]->FindBin(ij + 1), parserr[4]);

                  int ieta = (jk < 15) ? jk + 1 : 14 - jk;
                  const_eta_phi->Fill(ieta, ij + 1, calibc);
                  file_out << "intietax " << jk << " " << ij << " " << ieta << " "
                           << const_eta_phi->FindBin(ieta, ij + 1) << std::endl;
                  if (caliberr > 0) {
                    const_eta_phi->SetBinError(const_eta_phi->FindBin(ieta, ij + 1), caliberr);

                    mean_eta[ij] += calibc / (caliberr * caliberr);
                    mean_phi[jk] += calibc / (caliberr * caliberr);

                    rms_eta[ij] += 1. / (caliberr * caliberr);
                    rms_phi[jk] += 1. / (caliberr * caliberr);

                  } else {
                    const_eta_phi->SetBinError(const_eta_phi->FindBin(ieta, ij + 1), 0.0);
                  }
                } else if (iijj == 1) {
                  const_hpdrm[jk]->Fill(ij + 1, calibc);
                  const_hpdrm[jk]->SetBinError(const_hpdrm[jk]->FindBin(ij + 1), caliberr);

                  peak_hpdrm[jk]->Fill(ij + 1, fitres[4]);
                  peak_hpdrm[jk]->SetBinError(peak_hpdrm[jk]->FindBin(ij + 1), parserr[4]);
                }

                file_out << "HO  4 " << iijj << " " << std::setw(3) << kl << " " << std::setw(3) << ij + 1 << " "
                         << std::setw(7) << calibc << " " << std::setw(7) << caliberr << std::endl;
              }
            }

          } else {  //if (signall[izone]->GetEntries() >10) {
            signall[izone]->Draw();
            float varx = 0.000;
            int kl = (jk < 15) ? jk + 1 : 14 - jk;
            file_out << "histinfo" << iijj << " nof " << std::setw(3) << kl << " " << std::setw(3) << ij + 1 << " "
                     << std::setw(5) << pedstll[izone]->GetEntries() << " " << std::setw(6) << pedstll[izone]->GetMean()
                     << " " << std::setw(6) << pedstll[izone]->GetRMS() << " " << std::setw(5)
                     << signall[izone]->GetEntries() << " " << std::setw(6) << signall[izone]->GetMean() << " "
                     << std::setw(6) << signall[izone]->GetRMS() << " " << std::setw(6) << varx << " " << std::setw(3)
                     << varx << std::endl;

            file_out << "fitres x" << iijj << " " << kl << " " << ij + 1 << " " << varx << " " << varx << " " << varx
                     << " " << varx << " " << varx << " " << varx << " " << varx << std::endl;
            file_out << "parserr" << iijj << " " << kl << " " << ij + 1 << " " << varx << " " << varx << " " << varx
                     << " " << varx << " " << varx << " " << varx << " " << varx << std::endl;
          }
          iiter++;
          if (iiter % nsample == 0) {
            c0->Update();

            for (int kl = 0; kl < nsample; kl++) {
              if (gx0[kl]) {
                delete gx0[kl];
                gx0[kl] = nullptr;
              }
              if (ped0fun[kl]) {
                delete ped0fun[kl];
                ped0fun[kl] = nullptr;
              }
              if (signal[kl]) {
                delete signal[kl];
                signal[kl] = nullptr;
              }
              if (pedfun[kl]) {
                delete pedfun[kl];
                pedfun[kl] = nullptr;
              }
              if (sigfun[kl]) {
                delete sigfun[kl];
                sigfun[kl] = nullptr;
              }
              if (signalx[kl]) {
                delete signalx[kl];
                signalx[kl] = nullptr;
              }
              if (signall[kl]) {
                delete signall[kl];
                signall[kl] = nullptr;
              }
              if (pedstll[kl]) {
                delete pedstll[kl];
                pedstll[kl] = nullptr;
              }
            }
          }
        }  //for (int jk=0; jk<netamx; jk++) {
      }    //for (int ij=0; ij<nphimx; ij++) {

      //      if (iijj==0) {
      //	sprintf(out_file, "comb_hosig_allring_%i.jpg", irunold);
      //	c0->SaveAs(out_file);
      //	iiter = 0;
      //      } else {
      //	//	c0->Update();
      //      }

      //      iiter = 0;
    }  //end of iijj
    if (iiter % nsample != 0) {
      c0->Update();
      for (int kl = 0; kl < nsample; kl++) {
        if (gx0[kl]) {
          delete gx0[kl];
          gx0[kl] = nullptr;
        }
        if (ped0fun[kl]) {
          delete ped0fun[kl];
          ped0fun[kl] = nullptr;
        }
        if (signal[kl]) {
          delete signal[kl];
          signal[kl] = nullptr;
        }
        if (pedfun[kl]) {
          delete pedfun[kl];
          pedfun[kl] = nullptr;
        }
        if (sigfun[kl]) {
          delete sigfun[kl];
          sigfun[kl] = nullptr;
        }
        if (signalx[kl]) {
          delete signalx[kl];
          signalx[kl] = nullptr;
        }
        if (signall[kl]) {
          delete signall[kl];
          signall[kl] = nullptr;
        }
        if (pedstll[kl]) {
          delete pedstll[kl];
          pedstll[kl] = nullptr;
        }
      }
    }

    delete c0;

    xsiz = 600;  //int xsiz = 600;
    ysiz = 800;  //int ysiz = 800;

    gStyle->SetTitleFontSize(0.05);
    gStyle->SetTitleSize(0.025, "XYZ");
    gStyle->SetLabelSize(0.025, "XYZ");
    gStyle->SetStatFontSize(.045);

    gStyle->SetOptStat(0);
    ps.NewPage();
    TCanvas* c1 = new TCanvas("c1", " Pedestal vs signal", xsiz, ysiz);
    ped_evt->Draw();
    c1->Update();

    ps.NewPage();
    ped_statmean->Draw();
    c1->Update();

    ps.NewPage();
    ped_rms->Draw();
    c1->Update();

    ps.NewPage();
    ped_mean->Draw();
    c1->Update();

    ps.NewPage();
    ped_width->Draw();
    c1->Update();

    ps.NewPage();
    sig_evt->Draw();
    c1->Update();

    ps.NewPage();
    sig_statmean->Draw();
    c1->Update();

    ps.NewPage();
    sig_rms->Draw();
    c1->Update();

    ps.NewPage();
    fit_chi->Draw();
    c1->Update();

    ps.NewPage();
    fit_sigevt->Draw();
    c1->Update();

    ps.NewPage();
    fit_bkgevt->Draw();
    c1->Update();

    ps.NewPage();
    sig_mean->Draw();
    c1->Update();

    ps.NewPage();
    sig_width->Draw();
    c1->Update();

    ps.NewPage();
    sig_sigma->Draw();
    c1->Update();

    ps.NewPage();
    sig_meanerr->Draw();
    c1->Update();

    ps.NewPage();
    sig_meanerrp->Draw();
    c1->Update();

    ps.NewPage();
    sig_signf->Draw();
    c1->Update();

    ps.Close();
    delete c1;

    file_out.close();

    if (m_figure) {
      xsiz = 700;
      ysiz = 450;

      gStyle->SetTitleFontSize(0.09);
      gStyle->SetPadBottomMargin(0.17);
      gStyle->SetPadLeftMargin(0.18);
      gStyle->SetPadRightMargin(0.01);
      gStyle->SetOptLogy(0);
      gStyle->SetOptStat(0);

      TCanvas* c2 = new TCanvas("c2", "runfile", xsiz, ysiz);
      c2->Divide(5, 3);

      for (int side = 0; side < 2; side++) {
        gStyle->SetNdivisions(303, "XY");
        gStyle->SetPadRightMargin(0.01);
        int nmn = 0;
        int nmx = netamx / 2;
        if (side == 1) {
          nmn = netamx / 2;
          nmx = netamx;
        }

        int nzone = 0;

        for (int ij = nmn; ij < nmx; ij++) {
          c2->cd(nzone + 1);
          const_eta[ij]->GetXaxis()->SetTitle("#phi index");
          const_eta[ij]->GetXaxis()->SetTitleSize(.08);
          const_eta[ij]->GetXaxis()->CenterTitle();
          const_eta[ij]->GetXaxis()->SetTitleOffset(0.9);
          const_eta[ij]->GetXaxis()->SetLabelSize(.085);
          const_eta[ij]->GetXaxis()->SetLabelOffset(.01);

          const_eta[ij]->GetYaxis()->SetLabelSize(.08);
          const_eta[ij]->GetYaxis()->SetLabelOffset(.01);
          const_eta[ij]->GetYaxis()->SetTitle("GeV/MIP-GeV!!");

          const_eta[ij]->GetYaxis()->SetTitleSize(.085);
          const_eta[ij]->GetYaxis()->CenterTitle();
          const_eta[ij]->GetYaxis()->SetTitleOffset(1.3);
          const_eta[ij]->SetMarkerSize(0.60);
          const_eta[ij]->SetMarkerColor(2);
          const_eta[ij]->SetMarkerStyle(20);

          const_eta[ij]->Draw();
          nzone++;
        }

        sprintf(out_file, "calibho_%i_side%i.eps", irunold, side);
        c2->SaveAs(out_file);

        sprintf(out_file, "calibho_%i_side%i.jpg", irunold, side);
        c2->SaveAs(out_file);

        nzone = 0;
        for (int ij = nmn; ij < nmx; ij++) {
          c2->cd(nzone + 1);
          peak_eta[ij]->GetXaxis()->SetTitle("#phi index");
          peak_eta[ij]->GetXaxis()->SetTitleSize(.08);
          peak_eta[ij]->GetXaxis()->CenterTitle();
          peak_eta[ij]->GetXaxis()->SetTitleOffset(0.90);
          peak_eta[ij]->GetXaxis()->SetLabelSize(.08);
          peak_eta[ij]->GetXaxis()->SetLabelOffset(.01);

          peak_eta[ij]->GetYaxis()->SetLabelSize(.08);
          peak_eta[ij]->GetYaxis()->SetLabelOffset(.01);
          peak_eta[ij]->GetYaxis()->SetTitle("GeV");

          peak_eta[ij]->GetYaxis()->SetTitleSize(.085);
          peak_eta[ij]->GetYaxis()->CenterTitle();
          peak_eta[ij]->GetYaxis()->SetTitleOffset(1.3);

          peak_eta[ij]->SetMarkerSize(0.60);
          peak_eta[ij]->SetMarkerColor(2);
          peak_eta[ij]->SetMarkerStyle(20);

          peak_eta[ij]->Draw();
          nzone++;
        }

        sprintf(out_file, "peakho_%i_side%i.eps", irunold, side);
        c2->SaveAs(out_file);

        sprintf(out_file, "peakho_%i_side%i.jpg", irunold, side);
        c2->SaveAs(out_file);
      }
      delete c2;

      //      if (m_combined) {
      gStyle->SetTitleFontSize(0.045);
      gStyle->SetPadRightMargin(0.13);
      gStyle->SetPadBottomMargin(0.15);
      gStyle->SetPadLeftMargin(0.1);
      gStyle->SetOptStat(0);
      xsiz = 700;
      ysiz = 600;
      TCanvas* c1 = new TCanvas("c1", "Fitted const in each tower", xsiz, ysiz);
      const_eta_phi->GetXaxis()->SetTitle("#eta");
      const_eta_phi->GetXaxis()->SetTitleSize(0.065);
      const_eta_phi->GetXaxis()->SetTitleOffset(0.85);  //6);
      const_eta_phi->GetXaxis()->CenterTitle();
      const_eta_phi->GetXaxis()->SetLabelSize(0.045);
      const_eta_phi->GetXaxis()->SetLabelOffset(0.01);

      const_eta_phi->GetYaxis()->SetTitle("#phi");
      const_eta_phi->GetYaxis()->SetTitleSize(0.075);
      const_eta_phi->GetYaxis()->SetTitleOffset(0.5);
      const_eta_phi->GetYaxis()->CenterTitle();
      const_eta_phi->GetYaxis()->SetLabelSize(0.045);
      const_eta_phi->GetYaxis()->SetLabelOffset(0.01);

      const_eta_phi->Draw("colz");
      sprintf(out_file, "high_hoconst_eta_phi_%i.jpg", irunold);
      c1->SaveAs(out_file);

      delete c1;

      for (int jk = 0; jk < netamx; jk++) {
        int ieta = (jk < 15) ? jk + 1 : 14 - jk;
        if (rms_phi[jk] > 0) {
          mean_phi_ave->Fill(ieta, mean_phi[jk] / rms_phi[jk]);
          mean_phi_ave->SetBinError(mean_phi_ave->FindBin(ieta), pow(double(rms_phi[jk]), -0.5));
        }
      }

      for (int ij = 0; ij < nphimx; ij++) {
        if (rms_eta[ij] > 0) {
          mean_eta_ave->Fill(ij + 1, mean_eta[ij] / rms_eta[ij]);
          mean_eta_ave->SetBinError(mean_eta_ave->FindBin(ij + 1), pow(double(rms_eta[ij]), -0.5));
        }
      }

      ysiz = 450;
      gStyle->SetPadLeftMargin(0.13);
      gStyle->SetPadRightMargin(0.03);

      TCanvas* c2y = new TCanvas("c2", "Avearge signal in eta and phi", xsiz, ysiz);
      c2y->Divide(2, 1);
      mean_eta_ave->GetXaxis()->SetTitle("#phi");
      mean_eta_ave->GetXaxis()->SetTitleSize(0.085);
      mean_eta_ave->GetXaxis()->SetTitleOffset(0.65);
      mean_eta_ave->GetXaxis()->CenterTitle();
      mean_eta_ave->GetXaxis()->SetLabelSize(0.05);
      mean_eta_ave->GetXaxis()->SetLabelOffset(0.001);

      mean_eta_ave->GetYaxis()->SetTitle("Signal (GeV)/MIP");
      mean_eta_ave->GetYaxis()->SetTitleSize(0.055);
      mean_eta_ave->GetYaxis()->SetTitleOffset(1.3);
      mean_eta_ave->GetYaxis()->CenterTitle();
      mean_eta_ave->GetYaxis()->SetLabelSize(0.045);
      mean_eta_ave->GetYaxis()->SetLabelOffset(0.01);
      mean_eta_ave->SetMarkerSize(0.60);
      mean_eta_ave->SetMarkerColor(2);
      mean_eta_ave->SetMarkerStyle(20);

      c2y->cd(1);
      mean_eta_ave->Draw();

      mean_phi_ave->GetXaxis()->SetTitle("#eta");
      mean_phi_ave->GetXaxis()->SetTitleSize(0.085);
      mean_phi_ave->GetXaxis()->SetTitleOffset(0.65);  //55);
      mean_phi_ave->GetXaxis()->CenterTitle();
      mean_phi_ave->GetXaxis()->SetLabelSize(0.05);
      mean_phi_ave->GetXaxis()->SetLabelOffset(0.001);

      mean_phi_ave->GetYaxis()->SetTitle("Signal (GeV)/MIP");
      mean_phi_ave->GetYaxis()->SetTitleSize(0.055);
      mean_phi_ave->GetYaxis()->SetTitleOffset(1.3);
      mean_phi_ave->GetYaxis()->CenterTitle();
      mean_phi_ave->GetYaxis()->SetLabelSize(0.045);
      mean_phi_ave->GetYaxis()->SetLabelOffset(0.01);
      mean_phi_ave->SetMarkerSize(0.60);
      mean_phi_ave->SetMarkerColor(2);
      mean_phi_ave->SetMarkerStyle(20);

      c2y->cd(2);
      mean_phi_ave->Draw();

      sprintf(out_file, "high_hoaverage_eta_phi_%i.jpg", irunold);
      c2y->SaveAs(out_file);

      delete c2y;
      //      } else { //m_combined

      xsiz = 800;
      ysiz = 450;
      TCanvas* c3 = new TCanvas("c3", "Avearge signal in eta and phi", xsiz, ysiz);
      c3->Divide(2, 1);
      mean_phi_hst->GetXaxis()->SetTitle("#eta");
      mean_phi_hst->GetXaxis()->SetTitleSize(0.065);
      mean_phi_hst->GetXaxis()->SetTitleOffset(0.9);
      mean_phi_hst->GetXaxis()->CenterTitle();
      mean_phi_hst->GetXaxis()->SetLabelSize(0.065);
      mean_phi_hst->GetXaxis()->SetLabelOffset(0.001);

      mean_phi_hst->GetYaxis()->SetTitle("GeV/MIP");
      mean_phi_hst->GetYaxis()->SetTitleSize(0.055);
      mean_phi_hst->GetYaxis()->SetTitleOffset(0.9);
      mean_phi_hst->GetYaxis()->CenterTitle();
      mean_phi_hst->GetYaxis()->SetLabelSize(0.065);
      mean_phi_hst->GetYaxis()->SetLabelOffset(0.01);

      mean_phi_hst->SetMarkerColor(4);
      mean_phi_hst->SetMarkerSize(0.8);
      mean_phi_hst->SetMarkerStyle(20);
      mean_phi_hst->Draw();

      sprintf(out_file, "low_mean_phi_hst_%i.jpg", irunold);
      c3->SaveAs(out_file);

      delete c3;

      //      } //m_combined

      gStyle->SetOptLogy(1);
      gStyle->SetPadTopMargin(.1);
      gStyle->SetPadLeftMargin(.15);
      xsiz = 800;
      ysiz = 500;
      TCanvas* c0x = new TCanvas("c0x", "Signal in each ring", xsiz, ysiz);

      c0x->Divide(3, 2);
      for (int ij = 0; ij < ringmx; ij++) {
        int iread = (ij == 2) ? routmx : rout12mx;
        com_sigrsg[ij][iread]->GetXaxis()->SetTitle("Signal/ped (GeV)");

        com_sigrsg[ij][iread]->GetXaxis()->SetTitleSize(0.060);
        com_sigrsg[ij][iread]->GetXaxis()->SetTitleOffset(1.05);
        com_sigrsg[ij][iread]->GetXaxis()->CenterTitle();
        com_sigrsg[ij][iread]->GetXaxis()->SetLabelSize(0.065);
        com_sigrsg[ij][iread]->GetXaxis()->SetLabelOffset(0.01);

        com_sigrsg[ij][iread]->GetYaxis()->SetLabelSize(0.065);
        com_sigrsg[ij][iread]->GetYaxis()->SetLabelOffset(0.01);

        com_sigrsg[ij][iread]->SetLineWidth(3);
        com_sigrsg[ij][iread]->SetLineColor(4);

        c0x->cd(ij + 1);
        com_sigrsg[ij][iread]->Draw();

        com_crossg[ij][iread]->SetLineWidth(2);
        com_crossg[ij][iread]->SetLineColor(2);
        com_crossg[ij][iread]->Draw("same");
      }
      sprintf(out_file, "hosig_ring_%i.jpg", irunold);
      c0x->SaveAs(out_file);
      delete c0x;

      gStyle->SetTitleFontSize(0.06);
      gStyle->SetOptStat(0);
      gStyle->SetOptLogy(0);

      TCanvas* c0 = new TCanvas("c0", "Signal in each ring", xsiz, ysiz);

      c0->Divide(3, 2);
      for (int jk = 0; jk < ringmx; jk++) {
        peak_hpdrm[jk]->GetXaxis()->SetTitle("RM #");
        peak_hpdrm[jk]->GetXaxis()->SetTitleSize(0.070);
        peak_hpdrm[jk]->GetXaxis()->SetTitleOffset(1.0);
        peak_hpdrm[jk]->GetXaxis()->CenterTitle();
        peak_hpdrm[jk]->GetXaxis()->SetLabelSize(0.065);
        peak_hpdrm[jk]->GetXaxis()->SetLabelOffset(0.01);

        peak_hpdrm[jk]->GetYaxis()->SetTitle("Peak(GeV)/MIP");

        peak_hpdrm[jk]->GetYaxis()->SetTitleSize(0.07);
        peak_hpdrm[jk]->GetYaxis()->SetTitleOffset(1.3);
        peak_hpdrm[jk]->GetYaxis()->CenterTitle();
        peak_hpdrm[jk]->GetYaxis()->SetLabelSize(0.065);
        peak_hpdrm[jk]->GetYaxis()->SetLabelOffset(0.01);
        //	peak_hpdrm[jk]->SetLineWidth(3);
        //	peak_hpdrm[jk]->SetLineColor(4);
        peak_hpdrm[jk]->SetMarkerSize(0.60);
        peak_hpdrm[jk]->SetMarkerColor(2);
        peak_hpdrm[jk]->SetMarkerStyle(20);

        c0->cd(jk + 1);
        peak_hpdrm[jk]->Draw();
      }
      sprintf(out_file, "comb_peak_hpdrm_%i.jpg", irunold);
      c0->SaveAs(out_file);

      delete c0;

      TCanvas* c1y = new TCanvas("c1y", "Signal in each ring", xsiz, ysiz);

      c1y->Divide(3, 2);
      for (int jk = 0; jk < ringmx; jk++) {
        const_hpdrm[jk]->GetXaxis()->SetTitle("RM #");
        const_hpdrm[jk]->GetXaxis()->SetTitleSize(0.070);
        const_hpdrm[jk]->GetXaxis()->SetTitleOffset(1.3);
        const_hpdrm[jk]->GetXaxis()->CenterTitle();
        const_hpdrm[jk]->GetXaxis()->SetLabelSize(0.065);
        const_hpdrm[jk]->GetXaxis()->SetLabelOffset(0.01);

        const_hpdrm[jk]->GetYaxis()->SetTitle("Peak(GeV)");
        const_hpdrm[jk]->GetYaxis()->SetTitleSize(0.065);
        const_hpdrm[jk]->GetYaxis()->SetTitleOffset(1.0);
        const_hpdrm[jk]->GetYaxis()->CenterTitle();
        const_hpdrm[jk]->GetYaxis()->SetLabelSize(0.065);
        const_hpdrm[jk]->GetYaxis()->SetLabelOffset(0.01);
        //	const_hpdrm[jk]->SetLineWidth(3);
        //	const_hpdrm[jk]->SetLineColor(4);
        const_hpdrm[jk]->SetMarkerSize(0.60);
        const_hpdrm[jk]->SetMarkerColor(2);
        const_hpdrm[jk]->SetMarkerStyle(20);

        c1y->cd(jk + 1);
        const_hpdrm[jk]->Draw();
      }

      sprintf(out_file, "comb_const_hpdrm_%i.jpg", irunold);
      c1y->SaveAs(out_file);

      delete c1y;

    }  //if (m_figure) {

    //    ps.Close();
    //    file_out.close();

  }  // if (m_constant){

  if (m_figure) {
    for (int ij = 0; ij < nphimx; ij++) {
      for (int jk = 0; jk < netamx; jk++) {
        stat_eta[jk]->Fill(ij + 1, sigrsg[jk][ij]->GetEntries());
        statmn_eta[jk]->Fill(ij + 1, sigrsg[jk][ij]->GetMean());
      }
    }

    xsiz = 700;
    ysiz = 450;
    gStyle->SetTitleFontSize(0.09);
    gStyle->SetPadBottomMargin(0.14);
    gStyle->SetPadLeftMargin(0.17);
    gStyle->SetPadRightMargin(0.01);
    gStyle->SetNdivisions(303, "XY");
    gStyle->SetOptLogy(1);

    TCanvas* c2x = new TCanvas("c2x", "runfile", xsiz, ysiz);
    c2x->Divide(5, 3);
    for (int side = 0; side < 2; side++) {
      int nmn = 0;
      int nmx = netamx / 2;
      if (side == 1) {
        nmn = netamx / 2;
        nmx = netamx;
      }
      int nzone = 0;
      char name[200];

      for (int ij = nmn; ij < nmx; ij++) {
        int ieta = (ij < 15) ? ij + 1 : 14 - ij;
        c2x->cd(nzone + 1);
        sprintf(name, "GeV(#eta=%i)", ieta);
        sigrsg[ij][nphimx]->GetXaxis()->SetTitle(name);
        sigrsg[ij][nphimx]->GetXaxis()->SetTitleSize(.08);
        sigrsg[ij][nphimx]->GetXaxis()->CenterTitle();
        sigrsg[ij][nphimx]->GetXaxis()->SetTitleOffset(0.90);
        sigrsg[ij][nphimx]->GetXaxis()->SetLabelSize(.08);
        sigrsg[ij][nphimx]->GetXaxis()->SetLabelOffset(.01);

        sigrsg[ij][nphimx]->GetYaxis()->SetLabelSize(.08);
        sigrsg[ij][nphimx]->GetYaxis()->SetLabelOffset(.01);
        sigrsg[ij][nphimx]->SetLineWidth(2);
        sigrsg[ij][nphimx]->SetLineColor(4);
        sigrsg[ij][nphimx]->Draw();
        crossg[ij][nphimx]->SetLineWidth(2);
        crossg[ij][nphimx]->SetLineColor(2);
        crossg[ij][nphimx]->Draw("same");
        nzone++;
      }

      sprintf(out_file, "sig_ho_%i_side%i.eps", irunold, side);
      c2x->SaveAs(out_file);

      sprintf(out_file, "sig_ho_%i_side%i.jpg", irunold, side);
      c2x->SaveAs(out_file);
    }

    gStyle->SetOptLogy(0);
    c2x = new TCanvas("c2x", "runfile", xsiz, ysiz);
    c2x->Divide(5, 3);
    for (int side = 0; side < 2; side++) {
      int nmn = 0;
      int nmx = netamx / 2;
      if (side == 1) {
        nmn = netamx / 2;
        nmx = netamx;
      }
      int nzone = 0;

      nzone = 0;
      for (int ij = nmn; ij < nmx; ij++) {
        c2x->cd(nzone + 1);
        statmn_eta[ij]->SetLineWidth(2);
        statmn_eta[ij]->SetLineColor(4);
        statmn_eta[ij]->GetXaxis()->SetTitle("#phi index");
        statmn_eta[ij]->GetXaxis()->SetTitleSize(.08);
        statmn_eta[ij]->GetXaxis()->CenterTitle();
        statmn_eta[ij]->GetXaxis()->SetTitleOffset(0.9);
        statmn_eta[ij]->GetYaxis()->SetLabelSize(.08);
        statmn_eta[ij]->GetYaxis()->SetLabelOffset(.01);
        statmn_eta[ij]->GetXaxis()->SetLabelSize(.08);
        statmn_eta[ij]->GetXaxis()->SetLabelOffset(.01);
        statmn_eta[ij]->GetYaxis()->SetTitle("GeV");
        statmn_eta[ij]->GetYaxis()->SetTitleSize(.075);
        statmn_eta[ij]->GetYaxis()->CenterTitle();
        statmn_eta[ij]->GetYaxis()->SetTitleOffset(1.30);

        statmn_eta[ij]->Draw();
        nzone++;
      }

      sprintf(out_file, "statmnho_%i_side%i.eps", irunold, side);
      c2x->SaveAs(out_file);

      sprintf(out_file, "statmnho_%i_side%i.jpg", irunold, side);
      c2x->SaveAs(out_file);

      gStyle->SetOptLogy(1);
      gStyle->SetNdivisions(203, "XY");

      nzone = 0;
      for (int ij = nmn; ij < nmx; ij++) {
        c2x->cd(nzone + 1);
        stat_eta[ij]->SetLineWidth(2);
        stat_eta[ij]->SetLineColor(4);
        stat_eta[ij]->GetXaxis()->SetTitle("#phi index");
        stat_eta[ij]->GetXaxis()->SetTitleSize(.08);
        stat_eta[ij]->GetXaxis()->CenterTitle();
        stat_eta[ij]->GetXaxis()->SetTitleOffset(0.80);
        stat_eta[ij]->GetXaxis()->SetLabelSize(.08);
        stat_eta[ij]->GetXaxis()->SetLabelOffset(.01);
        stat_eta[ij]->GetYaxis()->SetLabelSize(.08);
        stat_eta[ij]->GetYaxis()->SetLabelOffset(.01);

        stat_eta[ij]->Draw();
        nzone++;
      }

      sprintf(out_file, "statho_%i_side%i.eps", irunold, side);
      c2x->SaveAs(out_file);

      sprintf(out_file, "statho_%i_side%i.jpg", irunold, side);
      c2x->SaveAs(out_file);
    }
    delete c2x;

  }  //if (m_figure) {

  if (!m_constant) {  //m_constant
    for (int jk = 0; jk < netamx; jk++) {
      for (int ij = 0; ij < nphimx; ij++) {
        if (crossg[jk][ij]) {
          delete crossg[jk][ij];
        }
        if (sigrsg[jk][ij]) {
          delete sigrsg[jk][ij];
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HOCalibAnalyzer);

/*
75minute 
112MB data
1M events

*/
