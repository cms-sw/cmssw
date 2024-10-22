#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TProfile.h"
#include "TPostScript.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TPaveStats.h"

#include "TRandom.h"

#include <string>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "TMinuit.h"
#include "TMath.h"

const char* prehist = "hoCalibc/";

using namespace std;
static unsigned int mypow_2[32];

int irunold = 1000;
const int nmxbin = 40000;
float yvalft[nmxbin];
float xvalft[nmxbin];
int nvalft = 0;

float alowx = -1.0;
float ahighx = 29.0;
int auprange = 25;

double fitchisq;
int fitndof;
double anormglb = -1;
Double_t gausX(Double_t* x, Double_t* par) { return par[0] * (TMath::Gaus(x[0], par[1], par[2], kTRUE)); }

// Double_t landX(Double_t* x, Double_t* par) {
//   return par[0]*(TMath::Landau(x[0], par[1], par[2]));
// }

// Double_t completefit(Double_t* x, Double_t* par) {
//   return gausX(x, par) + landX(x, &par[3]);
// }

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
  double scale = 1;          // par[1];
  double scale2 = anormglb;  // for notmalisation this is one, otehrwise use the normalisation;
  if (scale2 < .1)
    scale2 = 0.1;
  //  double scale=par[1];
  // Range of convolution integral
  xlow = x[0] - sc * scale * par[3];
  xupp = x[0] + sc * scale * par[3];

  step = (xupp - xlow) / np;

  // Convolution integral of Landau and Gaussian by sum
  for (double ij = 1.0; ij <= np / 2; ij++) {
    xx = xlow + (ij - .5) * step;
    fland = TMath::Landau(xx, mpc, par[0] * par[1], kTRUE);  // / par[0];
    if (xx > par[1]) {
      fland *= exp(-(xx - par[1]) / par[4]);
    }
    sum += fland * TMath::Gaus(x[0], xx, scale * par[3]);
    xx = xupp - (ij - .5) * step;
    fland = TMath::Landau(xx, mpc, par[0] * par[1], kTRUE);  // / par[0];
    if (xx > par[1]) {
      fland *= exp(-(xx - par[1]) / par[4]);
    }
    sum += fland * TMath::Gaus(x[0], xx, scale * par[3]);
  }
  return (par[2] * step * sum * invsq2pi / (scale2 * par[3]));
}

Double_t totalfunc(Double_t* x, Double_t* par) {
  return gausX(x, par) + langaufun(x, &par[3]);  // /max(.001,anormglb);
}

const int netamx = 30;
const int nphimx = 72;

int getHOieta(int ij) { return (ij < netamx / 2) ? -netamx / 2 + ij : -netamx / 2 + ij + 1; }
int invert_HOieta(int ieta) { return (ieta < 0) ? netamx / 2 + ieta : netamx / 2 + ieta - 1; }

int ietafit;
int iphifit;

void fcnsg(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag) {
  double xval[2];

  anormglb = 1;
  double tmpnorm = 0;
  for (int ij = 0; ij < 600; ij++) {
    xval[0] = -5 + (ij + 0.5) * .1;
    tmpnorm += 0.1 * langaufun(xval, &par[3]) / par[5];
  }
  anormglb = tmpnorm;

  double fval = -(par[0] + par[5]);
  for (int ij = 0; ij < nvalft; ij++) {
    if (yvalft[ij] < 1 || xvalft[ij] > auprange)
      continue;
    xval[0] = xvalft[ij];
    fval += yvalft[ij] * log(max(1.e-30, totalfunc(xval, par)));
  }
  f = -fval;
}

int main() {
  int icol = 0;
  int ntotal = 0;
  int irebin = 1;
  int ifile = 0;
  for (int ij = 0; ij < 32; ij++) {
    mypow_2[ij] = (int)pow(float(2), ij);
  }

  int max_nEvents = 300000000;
  int ityp = 1;
  bool m_cosmic = false;
  bool m_histFill = true;
  bool m_treeFill = false;
  bool m_zeroField = false;
  float pival = acos(-1.);

  int nsize;
  char outfile[100];
  char outfilx[100];
  char nametag[100];
  char nametag2[100];
  char infile[200];
  char datafile[100];
  char histname[100];
  char name[100];
  char title[100];

  cout << "Give the value of rebinning: a nonzero positive integer" << endl;
  cin >> irebin;
  if (irebin < 1) {
    irebin = 1;
  }
  cout << "Give the upper range of signal for fit [5 - 29]" << endl;
  cin >> auprange;

  cout << "Give the histname, e.g., 2017b_v2 2018a_v3a" << endl;
  cin >> nametag2;

  int len = strlen(nametag2);
  nametag2[len] = '\0';

  sprintf(outfilx, "hist_hoprompt_%s.root", nametag2);
  len = strlen(outfilx);
  outfilx[len] = '\0';

  //  TFile* fileIn = new TFile("histalla_hcalcalho_2016bcde_v2a.root", "read");

  TFile* fileIn = new TFile(outfilx, "read");
  //  const char* nametag="2016e_ar1";
  //  sprintf(nametag, "2016b_ar%i", irebin);

  unsigned ievt, hoflag;
  int irun, ilumi, nprim, isect, isect2, ndof, nmuon;

  float pileup, trkdr, trkdz, trkvx, trkvy, trkvz, trkmm, trkth, trkph, chisq, therr, pherr, hodx, hody, hoang, htime,
      hosig[9], hocorsig[18], hocro, hbhesig[9], caloen[3];
  float momatho, tkpt03, ecal03, hcal03;
  float tmphoang;

  TTree* Tin;
  if (m_treeFill) {
    Tin = (TTree*)fileIn->Get("T1");

    Tin->SetBranchAddress("irun", &irun);
    Tin->SetBranchAddress("ievt", &ievt);

    Tin->SetBranchAddress("isect", &isect);
    Tin->SetBranchAddress("isect2", &isect2);
    Tin->SetBranchAddress("ndof", &ndof);
    Tin->SetBranchAddress("nmuon", &nmuon);

    Tin->SetBranchAddress("ilumi", &ilumi);
    if (!m_cosmic) {
      Tin->SetBranchAddress("pileup", &pileup);
      Tin->SetBranchAddress("nprim", &nprim);
      Tin->SetBranchAddress("tkpt03", &tkpt03);
      Tin->SetBranchAddress("ecal03", &ecal03);
      Tin->SetBranchAddress("hcal03", &hcal03);
    }

    Tin->SetBranchAddress("trkdr", &trkdr);
    Tin->SetBranchAddress("trkdz", &trkdz);

    Tin->SetBranchAddress("trkvx", &trkvx);
    Tin->SetBranchAddress("trkvy", &trkvy);
    Tin->SetBranchAddress("trkvz", &trkvz);
    Tin->SetBranchAddress("trkmm", &trkmm);
    Tin->SetBranchAddress("trkth", &trkth);
    Tin->SetBranchAddress("trkph", &trkph);

    Tin->SetBranchAddress("chisq", &chisq);
    Tin->SetBranchAddress("therr", &therr);
    Tin->SetBranchAddress("pherr", &pherr);
    Tin->SetBranchAddress("hodx", &hodx);
    Tin->SetBranchAddress("hody", &hody);
    Tin->SetBranchAddress("hoang", &hoang);

    Tin->SetBranchAddress("momatho", &momatho);
    Tin->SetBranchAddress("hoflag", &hoflag);
    Tin->SetBranchAddress("htime", &htime);
    Tin->SetBranchAddress("hosig", hosig);
    Tin->SetBranchAddress("hocro", &hocro);
    Tin->SetBranchAddress("hocorsig", hocorsig);
    Tin->SetBranchAddress("caloen", caloen);
  }

  sprintf(nametag, "%s_ar%i_float_par8_rng%i", nametag2, irebin, auprange);

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.06);
  latex.SetTextFont(22);
  latex.SetTextAlign(11);  // 11 left; // 21 centre, // (31); // align right, 22, 23, shift bottom

  //Related fitting, not for storing
  const int nsample = 16;  //# of signal plots in the .ps page
  TF1* pedfun[nsample] = {0};
  TF1* sigfun[nsample] = {0};
  TF1* signalx[nsample] = {0};

  TH1F* signall[nsample] = {0};

  TH1F* signallunb[nsample] = {0};

  const int nbgpr = 3;
  const int nsgpr = 8;
  double fitprm[nsgpr][netamx];
  double xmn = -1.0;
  double xmx = 29.0;

  sprintf(outfilx, "fit_%s", nametag);
  len = strlen(outfilx);
  outfilx[len] = '\0';

  sprintf(outfile, "%s.txt", outfilx);
  ofstream file_out(outfile);

  sprintf(outfile, "%s.root", outfilx);

  TFile* fileOut = new TFile(outfile, "recreate");

  TTree* Tout;

  TH1F* sigrsg[netamx][nphimx];

  TH1F* fit_chi;
  TH1F* sig_evt;
  TH1F* fit_sigevt;
  TH1F* fit_bkgevt;
  TH1F* sig_mean;
  TH1F* sig_diff;
  TH1F* sig_width;
  TH1F* sig_sigma;
  TH1F* sig_expo;
  TH1F* sig_meanerr;
  TH1F* sig_meanerrp;
  TH1F* sig_signf;

  TH1F* sig_statmean;
  TH1F* sig_rms;

  TH2F* ped2d_evt;
  TH2F* ped2d_mean;
  TH2F* ped2d_width;
  TH2F* fit2d_chi;
  TH2F* sig2d_evt;
  TH2F* fit2d_sigevt;
  TH2F* fit2d_bkgevt;
  TH2F* sig2d_mean;
  TH2F* sig2d_diff;
  TH2F* sig2d_width;
  TH2F* sig2d_sigma;
  TH2F* sig2d_expo;
  TH2F* sig2d_meanerr;
  TH2F* sig2d_meanerrp;
  TH2F* sig2d_signf;
  TH2F* sig2d_rms;
  TH2F* sig2d_statmean;

  fit_chi = new TH1F("fit_chi", "fit_chi", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_evt = new TH1F("sig_evt", "sig_evt", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  fit_sigevt = new TH1F("fit_sigevt", "fit_sigevt", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  fit_bkgevt = new TH1F("fit_bkgevt", "fit_bkgevt", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_mean = new TH1F("sig_mean", "sig_mean", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_diff = new TH1F("sig_diff", "sig_diff", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_width = new TH1F("sig_width", "sig_width", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_sigma = new TH1F("sig_sigma", "sig_sigma", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_expo = new TH1F("sig_expo", "sig_expo", netamx * nphimx, -0.5, netamx * nphimx - 0.5);

  sig_meanerr = new TH1F("sig_meanerr", "sig_meanerr", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_meanerrp = new TH1F("sig_meanerrp", "sig_meanerrp", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_signf = new TH1F("sig_signf", "sig_signf", netamx * nphimx, -0.5, netamx * nphimx - 0.5);

  sig_statmean = new TH1F("sig_statmean", "sig_statmean", netamx * nphimx, -0.5, netamx * nphimx - 0.5);
  sig_rms = new TH1F("sig_rms", "sig_rms", netamx * nphimx, -0.5, netamx * nphimx - 0.5);

  fit2d_chi =
      new TH2F("fit2d_chi", "fit2d_chi", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_evt =
      new TH2F("sig2d_evt", "sig2d_evt", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  fit2d_sigevt = new TH2F(
      "fit2d_sigevt", "fit2d_sigevt", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  fit2d_bkgevt = new TH2F(
      "fit2d_bkgevt", "fit2d_bkgevt", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_mean = new TH2F(
      "sig2d_mean", "sig2d_mean", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_diff = new TH2F(
      "sig2d_diff", "sig2d_diff", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_width = new TH2F(
      "sig2d_width", "sig2d_width", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_sigma = new TH2F(
      "sig2d_sigma", "sig2d_sigma", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_expo = new TH2F(
      "sig2d_expo", "sig2d_expo", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);

  sig2d_statmean = new TH2F(
      "sig2d_statmean", "sig2d_statmean", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_rms =
      new TH2F("sig2d_rms", "sig2d_rms", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);

  sig2d_meanerr = new TH2F(
      "sig2d_meanerr", "sig2d_meanerr", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_meanerrp = new TH2F(
      "sig2d_meanerrp", "sig2d_meanerrp", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);
  sig2d_signf = new TH2F(
      "sig2d_signf", "sig2d_signf", netamx + 1, -netamx / 2. - 0.5, netamx / 2. + 0.5, nphimx, 0.5, nphimx + 0.5);

  if (m_histFill) {
    for (int jk = 0; jk < netamx; jk++) {
      for (int ij = 0; ij < nphimx; ij++) {
        //				sprintf(name, "%sho_indenergy_%i_%i", prehist, jk, ij);
        sprintf(name, "%ssig_eta%i_phi%i", prehist, jk, ij);
        sigrsg[jk][ij] = (TH1F*)fileIn->Get(name);
        cout << "jkij " << jk << " " << ij << " " << name << endl;
      }
    }
  } else if (m_treeFill) {
    for (int jk = 0; jk < netamx; jk++) {
      int ieta = getHOieta(jk);
      for (int ij = 0; ij < nphimx; ij++) {
        sprintf(name, "ho_indenergy_%i_%i", jk, ij);
        sprintf(title, "signal (i#eta=%i - i#phi=%i", ieta, ij + 1);
        sigrsg[jk][ij] = new TH1F(name, name, 120, -1.0, 14.0);  //1200, -1.0, 29.0);
      }
    }

    int nentries = Tin->GetEntries();
    for (int iev = 0; iev < nentries; iev++) {
      fileIn->cd();
      Tin->GetEntry(iev);
      fileOut->cd();
      int ieta = int((abs(isect) % 10000) / 100.) - 50;
      int iphi = abs(isect) % 100;
      int tmpxet = invert_HOieta(ieta);

      //event selection
      if (hosig[4] > -90) {
        sigrsg[tmpxet][iphi - 1]->Fill(hosig[4]);
      }
    }
  } else {
    cout << " You must read either histogramme or tree " << endl;
    return 1;
  }

  gStyle->SetTitleFontSize(0.075);
  gStyle->SetTitleBorderSize(1);
  gStyle->SetPadTopMargin(0.12);
  gStyle->SetPadBottomMargin(0.10);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetPadRightMargin(0.03);

  gStyle->SetOptStat(1110);
  gStyle->SetLabelSize(0.095, "XYZ");
  gStyle->SetLabelOffset(-0.01, "XYZ");
  gStyle->SetHistLineColor(1);
  gStyle->SetHistLineWidth(2);
  gStyle->SetPadGridX(0);
  gStyle->SetPadGridY(0);
  gStyle->SetGridStyle(0);
  gStyle->SetOptLogy(1);

  int ips = 111;
  sprintf(outfile, "%s.ps", outfilx);
  TPostScript ps(outfile, ips);
  ps.Range(20, 28);
  gStyle->SetOptLogy(0);
  gStyle->SetPadLeftMargin(0.17);

  gStyle->SetPadGridX(3);
  gStyle->SetPadGridY(3);
  gStyle->SetGridStyle(2);
  gStyle->SetPadRightMargin(0.17);
  gStyle->SetPadLeftMargin(0.10);

  gStyle->SetTitleFontSize(0.045);
  gStyle->SetPadTopMargin(0.10);     //.12
  gStyle->SetPadBottomMargin(0.12);  //.14
  gStyle->SetPadLeftMargin(0.17);
  gStyle->SetPadRightMargin(0.03);

  gStyle->SetOptStat(0);  //GMA (1110);

  gStyle->SetOptFit(101);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetStatBorderSize(1);
  gStyle->SetStatStyle(1001);
  gStyle->SetTitleColor(10);
  gStyle->SetTitleFontSize(0.09);
  gStyle->SetTitleOffset(-0.05);
  gStyle->SetTitleFillColor(10);
  gStyle->SetTitleBorderSize(1);

  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetStatColor(10);
  //    gStyle->SetStatFontSize(.07);
  gStyle->SetStatX(0.99);
  gStyle->SetStatY(0.99);
  gStyle->SetStatW(0.44);
  gStyle->SetStatH(0.16);
  gStyle->SetTitleSize(0.065, "XYZ");
  gStyle->SetLabelSize(0.075, "XYZ");
  gStyle->SetLabelOffset(0.012, "XYZ");
  gStyle->SetPadGridX(0);  //(1)
  gStyle->SetPadGridY(0);  //(1)
  gStyle->SetGridStyle(3);
  gStyle->SetNdivisions(101, "XY");
  gStyle->SetOptLogy(1);  //0); //GMA 1
  int iiter = 0;

  ps.NewPage();

  int xsiz = 900;   //900;
  int ysiz = 1200;  //600;
  TCanvas* c0 = new TCanvas("c0", " Pedestal vs signal", xsiz, ysiz);
  c0->Divide(4, 4);
  fileOut->cd();
  for (int ij = 0; ij < nphimx; ij++) {
    int iphi = ij + 1;
    //		for (int jk=0; jk<netamx; jk++) {
    for (int jk = 7; jk < 8; jk++) {
      int ieta = getHOieta(jk);
      int izone = iiter % nsample;

      signall[izone] = (TH1F*)sigrsg[jk][ij]->Clone("hnew");
      if (irebin > 1) {
        signall[izone]->Rebin(irebin);
      }

      signallunb[izone] = (TH1F*)signall[izone]->Clone("hnew");
      signall[izone]->Rebin(10 / irebin);

      if (izone == 0) {  //iiter%8 ==0) {
        ps.NewPage();
      }
      c0->cd(izone + 1);
      if (signall[izone]->GetEntries() / 2. > 5) {
        double binwid = signall[izone]->GetBinWidth(1);

        Double_t parall[nsgpr];
        double parserr[nsgpr];
        double fitres[nsgpr];
        double pedht = 0;

        char temp[20];
        xmn = signall[izone]->GetXaxis()->GetXmin();
        xmx = signall[izone]->GetXaxis()->GetXmax();
        int nbn = signall[izone]->FindBin(0);

        cout << "bincenter ===================================== " << signall[izone]->GetBinCenter(nbn) << endl;
        pedht = (signall[izone]->GetBinContent(nbn - 1) + signall[izone]->GetBinContent(nbn) +
                 signall[izone]->GetBinContent(nbn + 1)) /
                3.;

        parall[0] = max(1.0, 0.9 * pedht);               //Pedestal peak
        parall[1] = 0.00;                                //pedestal mean
        parall[2] = 0.03;                                //pedestal width
        parall[3] = 0.135;                               //Gaussian smearing of Landau function
        parall[4] = 0.7 * signallunb[izone]->GetMean();  //fitprm[4][jk]; //from 2015 cosmic data
        parall[5] = signallunb[izone]->GetEntries() / 2.;
        parall[6] = 0.2238;  // from 2015 cosmic data
        parall[7] = 5.0;

        nvalft = min(nmxbin, signallunb[izone]->GetNbinsX());

        for (int lm = 0; lm < nvalft; lm++) {
          xvalft[lm] = signallunb[izone]->GetBinCenter(lm + 1);
          yvalft[lm] = signallunb[izone]->GetBinContent(lm + 1);
        }

        TMinuit* gMinuit = new TMinuit(nsgpr);
        TString namex[nsgpr] = {"const", "mean", "sigma", "Width", "MP", "Area", "GSigma", "Exp"};
        double strt[nsgpr] = {parall[0], parall[1], parall[2], parall[3], parall[4], parall[5], parall[6], parall[7]};
        double alx = max(0.0, 0.1 * parall[0] - 1.1);
        double alowmn[nsgpr] = {
            alx, -0.1, -0.1, 0.06, 0.5 * strt[4] - 0.5, 0.1 * strt[5], 0.1 * strt[6], 0.5 * parall[7]};
        double ahighmn[nsgpr] = {
            5.0 * parall[0] + 10.1, 0.1, 0.1, 0.25, 1.5 * strt[4] + 0.5, 1.5 * strt[5], 3.2 * strt[6], 10.0 * parall[7]};

        double step[nsgpr] = {0.1, 0.01, 0.01, 0.001, 0.001, 1.0, 0.001, 0.01};

        gMinuit->SetFCN(fcnsg);

        double arglist[10];
        int ierflg = 0;
        arglist[0] = 0.5;
        gMinuit->mnexcm("SET ERR", arglist, 1, ierflg);

        for (int lm = 0; lm < nsgpr; lm++) {
          gMinuit->mnparm(lm, namex[lm], strt[lm], step[lm], alowmn[lm], ahighmn[lm], ierflg);
        }

        arglist[0] = 0;
        gMinuit->mnexcm("MINIMIZE", arglist, 0, ierflg);

        arglist[0] = 0;
        gMinuit->mnexcm("IMPROVE", arglist, 0, ierflg);

        TString chnam;
        double parv, err, xlo, xup, plerr, mierr, eparab, gcc;
        int iuit;

        for (int lm = 0; lm < nsgpr; lm++) {
          gMinuit->mnpout(lm, chnam, parv, err, xlo, xup, iuit);
          gMinuit->mnerrs(lm, plerr, mierr, eparab, gcc);
          fitres[lm] = fitprm[lm][jk] = parv;
          parserr[lm] = err;
        }

        fitres[0] *= binwid;
        fitres[5] *= binwid;

        double fedm, errdef;
        int nparx, istat;
        gMinuit->mnstat(fitchisq, fedm, errdef, fitndof, nparx, istat);

        delete gMinuit;

        anormglb = 1;
        double tmpnorm = 0;
        for (int ix = 0; ix < 600; ix++) {
          double xval[2];
          xval[0] = -5 + (ix + 0.5) * .1;
          tmpnorm += 0.1 * langaufun(xval, &fitres[3]) / fitres[5];
        }
        anormglb = tmpnorm;

        double stp = 30 * fitres[3] * fitres[4] / 1000.;
        double str = fitres[4] * (1. - 5. * fitres[3]);
        double xx[2];
        double sum1 = 0;
        double sum2 = 0;

        for (int lm = 0; lm < 1000; lm++) {
          xx[0] = str + (lm + 0.5) * stp;
          double landf = langaufun(xx, &fitres[3]);  //No need of normalisation
          sum1 += landf;
          sum2 += xx[0] * landf;
        }

        sum2 /= TMath::Max(0.1, sum1);
        //	    signall[izone]->GetXaxis()->SetRangeUser(-0.25, 4.75);
        signall[izone]->Draw("hist");

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

        cout << "sum2 " << sum2 << " " << sigfun[izone]->Integral(fitres[4] * (1. - 5. * fitres[3]), str + 1000.5 * stp)
             << " " << binwid << endl;

        sprintf(temp, "total_%i", izone);
        signalx[izone] = new TF1(temp, totalfunc, xmn, xmx, nsgpr);
        signalx[izone]->SetParameters(fitres);
        signalx[izone]->SetLineWidth(1);
        signalx[izone]->Draw("same");

        latex.DrawLatex(
            0.60, 0.83, Form("#mu: %g#pm%g", int(1000 * fitres[4]) / 1000., int(1000 * parserr[4]) / 1000.));
        latex.DrawLatex(0.60,
                        0.765,
                        Form("#Gamma: %g#pm%g",
                             int(1000 * fitres[3] * fitres[4]) / 1000.,
                             int(1000 * (sqrt((fitres[3] * parserr[4]) * (fitres[3] * parserr[4]) +
                                              (fitres[4] * parserr[3]) * (fitres[4] * parserr[3])))) /
                                 1000.));
        latex.DrawLatex(
            0.60, 0.70, Form("#sigma: %g#pm%g", int(1000 * fitres[6]) / 1000., int(1000 * parserr[6]) / 1000.));
        latex.DrawLatex(0.65, 0.64, Form("A: %g#pm%g", int(1 * fitres[5]) / 1., int(10 * parserr[5]) / 10.));
        latex.DrawLatex(0.67, 0.58, Form("Ex: %g#pm%g", int(10 * fitres[7]) / 10., int(10 * parserr[7]) / 10.));

        latex.DrawLatex(0.67, 0.52, Form("Mean: %g ", int(100 * sum2) / 100.));

        cout << "histinfo fit " << std::setw(3) << ieta << " " << std::setw(3) << ij + 1 << " " << std::setw(5)
             << signallunb[izone]->GetEntries() / 2. << " " << std::setw(6) << signallunb[izone]->GetMean() << " "
             << std::setw(6) << signallunb[izone]->GetRMS() << " " << std::setw(6) << fitchisq << " " << std::setw(3)
             << fitndof << endl;

        file_out << "histinfo fit " << std::setw(3) << ieta << " " << std::setw(3) << ij + 1 << " " << std::setw(5)
                 << signallunb[izone]->GetEntries() / 2. << " " << std::setw(6) << signallunb[izone]->GetMean() << " "
                 << std::setw(6) << signallunb[izone]->GetRMS() << endl;

        file_out << "fitresx " << ieta << " " << ij + 1 << " " << fitres[0] / binwid << " " << fitres[1] << " "
                 << fitres[2] << " " << fitres[3] << " " << fitres[4] << " " << fitres[5] / binwid << " " << fitres[6]
                 << " " << signallunb[izone]->GetEntries() / 2. << " " << fitchisq << " " << fitndof << " " << binwid
                 << " " << sum2 << " " << fitres[7] << endl;
        file_out << "parserr " << ieta << " " << ij + 1 << " " << parserr[0] << " " << parserr[1] << " " << parserr[2]
                 << " " << parserr[3] << " " << parserr[4] << " " << parserr[5] << " " << parserr[6] << " "
                 << parserr[7] << endl;

        double diff = fitres[4] - fitres[1];
        if (diff <= 0)
          diff = 0.000001;

        int ifl = nphimx * jk + ij;

        fit_chi->Fill(ifl, fitchisq);  //signal[izone]->GetChisquare());
        sig_evt->Fill(ifl, signallunb[izone]->GetEntries() / 2.);
        fit_sigevt->Fill(ifl, fitres[5] / binwid);
        fit_bkgevt->Fill(ifl, fitres[0] / binwid);

        sig_mean->Fill(ifl, fitres[4]);
        sig_diff->Fill(ifl, fitres[4] - fitres[1]);
        sig_width->Fill(ifl, fitres[3]);
        sig_sigma->Fill(ifl, fitres[6]);
        sig_expo->Fill(ifl, fitres[7]);
        sig_meanerr->Fill(ifl, parserr[4]);
        if (fitres[4] - fitres[1] > 1.e-4)
          sig_meanerrp->Fill(ifl, 100 * parserr[4] / (fitres[4] - fitres[1]));
        if (fitres[2] > 1.e-4)
          sig_signf->Fill(ifl, (fitres[4] - fitres[1]) / fitres[2]);

        sig_statmean->Fill(ifl, signallunb[izone]->GetMean());
        sig_rms->Fill(ifl, signallunb[izone]->GetRMS());

        fit2d_chi->Fill(ieta, iphi, fitchisq);  //signal[izone]->GetChisquare());
        sig2d_evt->Fill(ieta, iphi, signallunb[izone]->GetEntries() / 2.);
        fit2d_sigevt->Fill(ieta, iphi, fitres[5] / binwid);
        fit2d_bkgevt->Fill(ieta, iphi, fitres[0] / binwid);

        sig2d_mean->Fill(ieta, iphi, fitres[4]);
        sig2d_diff->Fill(ieta, iphi, fitres[4] - fitres[1]);
        sig2d_width->Fill(ieta, iphi, fitres[3]);
        sig2d_sigma->Fill(ieta, iphi, fitres[6]);
        sig2d_expo->Fill(ieta, iphi, fitres[7]);

        sig2d_meanerr->Fill(ieta, iphi, parserr[4]);
        if (fitres[4] - fitres[1] > 1.e-4)
          sig2d_meanerrp->Fill(ieta, iphi, 100 * parserr[4] / (fitres[4] - fitres[1]));
        if (fitres[2] > 1.e-4)
          sig2d_signf->Fill(ieta, iphi, (fitres[4] - fitres[1]) / fitres[2]);

        sig2d_statmean->Fill(ieta, iphi, signallunb[izone]->GetMean());
        sig2d_rms->Fill(ieta, iphi, signallunb[izone]->GetRMS());
      } else {  //if (signallunb[izone]->GetEntries()/2. >10) {
        signall[izone]->Draw();
        float varx = 0.000;

        file_out << "histinfo nof " << std::setw(3) << ieta << " " << std::setw(3) << ij + 1 << " " << std::setw(5)
                 << signallunb[izone]->GetEntries() / 2. << " " << std::setw(6) << signallunb[izone]->GetMean() << " "
                 << std::setw(6) << signallunb[izone]->GetRMS() << endl;

        file_out << "fitresx " << ieta << " " << ij + 1 << " " << varx << " " << varx << " " << varx << " " << varx
                 << " " << varx << " " << varx << " " << varx << " " << varx << " " << varx << " " << varx << " "
                 << varx << " " << varx << " " << varx << endl;
        file_out << "parserr " << ieta << " " << ij + 1 << " " << varx << " " << varx << " " << varx << " " << varx
                 << " " << varx << " " << varx << " " << varx << " " << varx << " " << varx << endl;
      }
      iiter++;
      if (iiter % nsample == 0) {
        c0->Update();

        for (int lm = 0; lm < nsample; lm++) {
          if (pedfun[lm]) {
            delete pedfun[lm];
            pedfun[lm] = 0;
          }
          if (sigfun[lm]) {
            delete sigfun[lm];
            sigfun[lm] = 0;
          }
          if (signalx[lm]) {
            delete signalx[lm];
            signalx[lm] = 0;
          }
          if (signall[lm]) {
            delete signall[lm];
            signall[lm] = 0;
          }
          if (signallunb[lm]) {
            delete signallunb[lm];
            signallunb[lm] = 0;
          }
        }
      }
    }  //for (int jk=0; jk<netamx; jk++) {
  }    //for (int ij=0; ij<nphimx; ij++) {

  if (iiter % nsample != 0) {
    c0->Update();
    for (int lm = 0; lm < nsample; lm++) {
      if (pedfun[lm]) {
        delete pedfun[lm];
        pedfun[lm] = 0;
      }
      if (sigfun[lm]) {
        delete sigfun[lm];
        sigfun[lm] = 0;
      }
      if (signalx[lm]) {
        delete signalx[lm];
        signalx[lm] = 0;
      }
      if (signall[lm]) {
        delete signall[lm];
        signall[lm] = 0;
      }
      if (signallunb[lm]) {
        delete signallunb[lm];
        signallunb[lm] = 0;
      }
    }
  }

  if (c0) {
    delete c0;
    c0 = 0;
  }

  xsiz = 600;  //int xsiz = 600;
  ysiz = 800;  //int ysiz = 800;

  ps.NewPage();
  gStyle->SetOptLogy(0);
  gStyle->SetTitleFontSize(0.05);
  gStyle->SetTitleSize(0.025, "XYZ");
  gStyle->SetLabelSize(0.025, "XYZ");
  gStyle->SetStatFontSize(.045);

  gStyle->SetPadGridX(1);  //bool input yes/no, must give an input
  gStyle->SetPadGridY(1);
  gStyle->SetGridStyle(3);

  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(.07);
  gStyle->SetPadLeftMargin(0.07);

  ps.NewPage();
  TCanvas* c1 = new TCanvas("c1", " Pedestal vs signal", xsiz, ysiz);
  sig_evt->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_statmean->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_rms->Draw("hist");
  c1->Update();

  ps.NewPage();
  fit_chi->Draw("hist");
  c1->Update();

  ps.NewPage();
  fit_sigevt->Draw("hist");
  c1->Update();

  ps.NewPage();
  fit_bkgevt->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_mean->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_width->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_sigma->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_expo->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_meanerr->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_meanerrp->Draw("hist");
  c1->Update();

  ps.NewPage();
  sig_signf->Draw("hist");
  c1->Update();

  gStyle->SetPadLeftMargin(0.06);
  gStyle->SetPadRightMargin(0.15);

  ps.NewPage();
  TCanvas* c2y = new TCanvas("c2y", " Pedestal vs Signal", xsiz, ysiz);

  sig2d_evt->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_statmean->SetMaximum(min(3.0, sig2d_statmean->GetMaximum()));
  sig2d_statmean->SetMinimum(max(-2.0, sig2d_statmean->GetMinimum()));
  sig2d_statmean->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_rms->SetMaximum(min(3.0, sig2d_rms->GetMaximum()));
  sig2d_rms->SetMinimum(max(0.0, sig2d_rms->GetMinimum()));
  sig2d_rms->Draw("colz");
  c2y->Update();

  ps.NewPage();

  fit2d_chi->Draw("colz");
  c2y->Update();

  ps.NewPage();
  fit2d_sigevt->Draw("colz");
  c2y->Update();

  ps.NewPage();
  fit2d_bkgevt->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_mean->SetMaximum(min(2.0, sig2d_mean->GetMaximum()));
  sig2d_mean->SetMinimum(max(0.1, sig2d_mean->GetMinimum()));
  sig2d_mean->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_width->SetMaximum(min(0.5, sig2d_width->GetMaximum()));
  sig2d_width->SetMinimum(max(0.01, sig2d_width->GetMinimum()));
  sig2d_width->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_sigma->SetMaximum(min(0.5, sig2d_sigma->GetMaximum()));
  sig2d_sigma->SetMinimum(max(0.01, sig2d_sigma->GetMinimum()));
  sig2d_sigma->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_expo->SetMaximum(min(20., sig2d_expo->GetMaximum()));
  sig2d_expo->SetMinimum(max(2.0, sig2d_expo->GetMinimum()));
  sig2d_expo->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_meanerr->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_meanerrp->Draw("colz");
  c2y->Update();

  ps.NewPage();
  sig2d_signf->Draw("colz");
  c2y->Update();

  ps.Close();

  delete c1;
  delete c2y;

  file_out.close();

  fileOut->cd();
  fileOut->Write();
  fileOut->Close();

  fileIn->cd();
  fileIn->Close();
}
