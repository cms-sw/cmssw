/*
Identify digi and other correlaions
Increase range of Digi plots
Swap digi reco option
Noise : Second TS has sharp fall
Previous postion of SiPM : Problematic


hadd hist_hoprompt_r2017ae.root hist_r2017a_hoprompt_v2a_1.root hist_r2017a_hoprompt_v3a_1.root hist_hoprompt_r2017b_v1.root hist_r2017b_hoprompt_v2a_1.root hist_r2017c_hoprompt_v1a_1.root hist_hoprompt_r2017c_v2.root hist_hoprompt_r2017c_v3.root hist_hoprompt_r2017d_v1.root hist_hoprompt_r2017e_v1.root


*/
const int netamx = 30;
const int nphimx = 72;
int getieta(int ij) { return (ij < netamx / 2) ? -netamx / 2 + ij : -netamx / 2 + ij + 1; }
int invert_ieta(int ieta) { return (ieta < 0) ? netamx / 2 + ieta : netamx / 2 + ieta - 1; }

const int ncut = 14;
const int ringmx = 5;
const int nchnmx = 10;
const int routmx = 36;
const int rout12mx = 24;

const int rbxmx = 12;     // HO readout box in Ring 0
const int rbx12mx = 6;    //HO readout box in Ring+-1/2
const int nprojtype = 5;  // Varities due to types of muon projection
const char* projname[nprojtype] = {"Noise", "In RM", "3x3", "Proj", "Signal"};
const int nprojmx = 4;
const int nseltype = 4;  //Different crit for muon selection
const int nadmx = 18;
const int shapemx = 10;  //checking shape

static const int nhothresh = 10;  // set threshold for noise filter

TH2F* totalmuon;
TH2F* totalproj[9];
TH2F* totalhpd[18];

TH2F* totalprojsig[9];
TH2F* totalhpdsig[18];

TH2F* total2muon;
TH2F* total2proj[9];
TH2F* total2hpd[18];

TH2F* total2projsig[9];
TH2F* total2hpdsig[18];

Double_t gausX(Double_t* x, Double_t* par) { return par[0] * (TMath::Gaus(x[0], par[1], par[2], kTRUE)); }

Double_t landX(Double_t* x, Double_t* par) { return par[0] * (TMath::Landau(x[0], par[1], par[2])); }

Double_t completefit(Double_t* x, Double_t* par) { return gausX(x, par) + landX(x, &par[3]); }

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
  double scale = 1;   // par[1];
  double scale2 = 1;  //anormglb; // for notmalisation this is one, otehrwise use the normalisation;
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
// binroot hocalib_r2017e_hoprompt_v1a_1.root hocalib_r2017g_hoprompt_v1a_1.root hocalib_r2017h_hoprompt_v1a_1.root
void momentum() {
  gStyle->SetOptStat(1110);
  const int nfile = 3;
  TH1F* histx[nfile];
  TTree* T1x;
  char name[100];
  TCanvas* c1 = new TCanvas("c1", "runfile", 700, 300);
  c1->Divide(3, 1);

  for (int ij = 0; ij < nfile; ij++) {
    c1->cd(ij + 1);
    sprintf(name, "hist_%i", ij);
    histx[ij] = new TH1F(name, name, 120, -150., 150.);
    if (ij == 0) {
      //      _file2->cd(); // T1x = (TTree*)_file0->Get("T1");
    } else if (ij == 1) {
      //      _file0->cd(); // T1x = (TTree*)_file1->Get("T1");
    } else {
      //      _file2->cd();// T1x = (TTree*)_file2->Get("T1");
    }
    switch (ij) {
      case 0:
        _file0->cd();
        break;
      case 1:
        _file1->cd();
        break;
      case 2:
        _file2->cd();
        break;
    }

    T1x = (TTree*)gDirectory->Get("T1");
    T1x->Project(name, "trkmm");
    //    histx[ij]->Scale(1./TMath::Max(1.,histx[ij]->Integral()));
    histx[ij]->Draw();
    //    T1x->Draw("trkmm");
  }
}

/*
binroot histalla_hcalcalho_2016btog.root
 .L tdrstyle.C 
setTDRStyle()
.L hocalib_pl16.C
plot_fitres(23, 42);


*/

void plot_fitres(int ieta = 13, int iphi = 70) {
  gStyle->SetOptTitle(0);
  gStyle->SetOptLogy(1);
  //   gStyle->SetOptStat(0); //1110);
  //   gStyle->SetLabelSize(0.095,"XYZ");
  //   gStyle->SetLabelOffset(0.01,"XYZ");
  //   gStyle->SetHistLineColor(1);
  //   gStyle->SetHistLineWidth(3);
  //   gStyle->SetPadTopMargin(0.02);
  //   gStyle->SetPadBottomMargin(0.12);
  //   gStyle->SetPadLeftMargin(0.14);
  //   gStyle->SetPadRightMargin(0.04);
  //   gStyle->SetPadGridX(3);
  //   gStyle->SetPadGridY(3);
  //   gStyle->SetGridStyle(2);
  //   gStyle->SetMarkerColor(1);
  //   gStyle->SetMarkerStyle(20);
  //   gStyle->SetMarkerSize(0.95);
  //   gStyle->SetNdivisions(506,"XY");
  //   gStyle->SetLabelSize(0.095,"XYZ");
  //   gStyle->SetLabelOffset(0.01,"XYZ");
  //   gStyle->SetStatH(.1);
  //   gStyle->SetStatW(.4);

  // -14, 1 plot_fitres(1,0);
  //  double par[8]={139.1, 0.0327856, 0.070529, 0.137647, 1.69858, 79549.1, 0.307275, 5.71755};

  // -8, 1 plot_fitres(7,0);
  //  double par[8]={31.5185, 0.0342954, 0.0599271, 0.151538, 1.10591, 91530.5, 0.22909, 8.65218};
  // -1, 1 plot_fitres(14,0);
  double par[8] = {23.6947, 0.0178834, 0.0318413, 0.160218, 0.985584, 77652.5, 0.15613, 10.0053};

  //-15, 1
  //  double par[8]={89.055, 0.0277592, 0.057265, 0.106515, 1.80511, 36326, 0.36995, 6.04276};
  // 9, 43  plot_fitres(23,42)
  //  double par[8]={ 29.3954, 0.0339121, 0.0644736, 0.160249, 1.15268, 94723.6, 0.239194, 10.5442};
  // 10, 43 plot_fitres(24,42)
  //  double par[8]={0.100621, 0.0413684, 0.077322, 0.154082, 1.22422, 80333, 0.315477, 12.1145};
  // 11, 43 plot_fitres(25,42)
  //  double par[8]={0.91, 0.0523604, 0.1, 0.25, 1.24822, 71042.8, 0.373684, 5.42163};
  // 12, 43 plot_fitres(26,42)
  //  double par[8]={0.550151, 0.0584881, 0.1, 0.246599, 1.38702, 84966.5, 0.35758, 4.92687};

  par[0] /= 18.0;
  par[5] /= 18.0;

  const char title[100];
  TH1F* histx[2];
  TH1F* histy[2];

  TF1* ped0fun;
  TF1* pedfun;
  TF1* sigfun;
  TF1* signalx;
  double xmn = -15.;
  double xmx = 20.;

  TCanvas* c1 = new TCanvas("c1", "runfile", 600, 600);
  //  c1->Divide(2,1);
  for (int i = 1; i < 2; i++) {
    //    c1->cd(i+1);
    switch (i) {
      case 0:
        sprintf(title, "hoCalibc/ped_eta%i_phi%i", ieta, iphi);
        break;
      case 1:
        sprintf(title, "hoCalibc/sig_eta%i_phi%i", ieta, iphi);
        break;
    }

    histy[i] = (TH1F*)gDirectory->Get(title);
    histx[i] = (TH1F*)histy[i]->Clone();
    cout << "name " << title << " " << histx[i]->GetBinWidth(100) << " " << histx[i]->GetTitle() << endl;
    //    histx[i]->GetXaxis()->SetRangeUser(xmn, (i+1)*xmx);
    histx[i]->SetLineColor(2 * (i + 1));
    histx[i]->SetLineWidth(2);
    histx[i]->Rebin(10);

    histx[i]->SetMaximum(1.52 * histx[i]->GetMaximum());

    //    histx[i]->SetNdivisions(506,"XY");

    //    histx[i]->GetXaxis()->SetLabelSize(.055);
    //    histx[i]->GetYaxis()->SetLabelSize(.055);
    //    histx[i]->GetYaxis()->SetTitleOffset(1.4);

    switch (i) {
      case 0:
        histx[i]->GetXaxis()->SetRangeUser(-16., 16.);
        histx[i]->GetXaxis()->SetTitle("HO Pedestal (GeV)");
        break;
      case 1:
        histx[i]->GetXaxis()->SetRangeUser(-20., 40.);
        histx[i]->GetXaxis()->SetTitle("HO Signal (GeV)");
        break;
    }
    //    histx[i]->GetXaxis()->SetTitleSize(.055);
    //    //      histx[i]->GetXaxis()->CenterTitle();
    histx[i]->GetYaxis()->SetTitle("Nevents");
    //    histx[i]->GetYaxis()->SetTitleSize(.055);
    //      if (i==0) histx[i]->GetXaxis()->SetRangeUser(-5., 5.);
    histx[i]->Draw();
  }

  //   c1->cd(1);
  //   TPaveText *ptst1 = new TPaveText(.85,0.9,.90,.95,"brNDC");
  //   ptst1->SetFillColor(10);
  //   TText* text1 = ptst1->AddText("(a)");
  //   text1->SetTextSize(0.062);
  //   ptst1->SetBorderSize(0);
  //   ptst1->Draw();

  //   ped0fun = new TF1("temp", gausX, xmn, xmx, 3);
  //   ped0fun->SetParameters(gaupr);
  //   ped0fun->SetLineColor(3);
  //   ped0fun->SetLineWidth(2);
  //   ped0fun->Draw("same");

  //   c1->cd(2);

  //   TPaveText *ptst = new TPaveText(0.85,0.9,0.90,.95,"brNDC");
  //   ptst->SetFillColor(10);
  //   TText* text = ptst->AddText("(b)");
  //   text->SetTextSize(0.062);
  //   ptst->SetBorderSize(0);
  //   ptst->Draw();

  pedfun = new TF1("ped0", gausX, xmn, 4 * xmx, 3);
  pedfun->SetParameters(par);
  pedfun->SetLineWidth(2);
  pedfun->SetLineColor(3);
  pedfun->Draw("same");

  sigfun = new TF1("signalfun", langaufun, xmn, 4 * xmx, 5);
  sigfun->SetParameters(&par[3]);
  sigfun->SetLineWidth(2);
  sigfun->SetLineColor(4);
  sigfun->Draw("same");

  signalx = new TF1("total", totalfunc, xmn, 4 * xmx, 8);
  signalx->SetParameters(par);
  signalx->SetLineWidth(2);
  signalx->SetLineWidth(1);
  signalx->Draw("same");

  cmsPrel(.55, .75, .55, .75, 0.045, 35);
}

/*
 binroot fit_2016btog_ar1_float_par8_rng30.root
 .L tdrstyle.C 
setTDRStyle()
.L hocalib_pl16.C
const_term_1dx("const_eta_phi")
const_term_2d()
*/

/*
binroot fit_r2017a_v2_ar1_float_par8_rng25.root fit_r2017a_v3_ar1_float_par8_rng25.root fit_r2017b_v1_ar1_float_par8_rng25.root fit_r2017b_v2_ar1_float_par8_rng25.root fit_r2017c_v1_ar1_float_par8_rng25.root fit_r2017c_v2_ar1_float_par8_rng25.root fit_r2017c_v3_ar1_float_par8_rng25.root fit_r2017d_v1_ar1_float_par8_rng25.root fit_r2017e_v1_ar1_float_par8_rng25.root fit_r2017f_v1_ar1_float_par8_rng25.root fit_r2017g_v1_ar1_float_par8_rng25.root fit_r2017h_v1_ar1_float_par8_rng25.root fit_r2017ae_ar1_float_par8_rng25.root fit_r2017af_ar1_float_par8_rng25.root




*/

void all_1d4x() {
  TCanvas* c1 = new TCanvas("c1", "c1", 700, 500);
  c1->Divide(2, 2);

  _file0->cd();
  const_term_1d4x();
  _file1->cd();
  const_term_1d4x();
  _file2->cd();
  const_term_1d4x();
  _file3->cd();
  const_term_1d4x();
  _file4->cd();
  const_term_1d4x();
  _file5->cd();
  const_term_1d4x();
  _file6->cd();
  const_term_1d4x();
  _file7->cd();
  const_term_1d4x();
  _file8->cd();
  const_term_1d4x();
  _file9->cd();
  const_term_1d4x();
  _file10->cd();
  const_term_1d4x();
  _file11->cd();
  const_term_1d4x();
  _file12->cd();
  const_term_1d4x();
  _file13->cd();
  const_term_1d4x();

  /*
fit_r2017a_v2_a
val   0.363+- 0.294  0.211+- 0.173  0.468+- 0.184  0.282+-  0.11
err   5.461+-  0.27  4.622+- 0.155  4.444+- 0.177  4.765+- 0.092
fit_r2017a_v3_a
val   0.554+- 0.105 -0.372+- 0.073 -0.165+-  0.08 -0.063+- 0.047
err   2.205+- 0.096  2.039+-  0.06  2.029+- 0.066  2.159+- 0.037
fit_r2017b_v1_a
val   1.577+- 0.043  0.964+- 0.032  1.234+- 0.035  1.208+- 0.021
err   0.944+- 0.036    0.9+- 0.027  0.934+- 0.028  0.974+- 0.018
fit_r2017b_v2_a
val   2.209+- 0.072  1.826+-  0.05  2.038+- 0.055  1.969+- 0.032
err    1.54+- 0.073   1.37+- 0.039  1.397+- 0.047  1.485+- 0.026
fit_r2017c_v1_a
val   2.439+- 0.056  1.933+-  0.04  2.125+- 0.046  2.118+- 0.027
err   1.208+-  0.05   1.12+- 0.032  1.225+- 0.038   1.22+- 0.022
fit_r2017c_v2_a
val   2.569+- 0.045  1.996+- 0.032  2.289+- 0.038  2.247+- 0.022
err   0.969+- 0.035   0.93+- 0.028   0.97+- 0.034  0.995+- 0.018
fit_r2017c_v3_a
val   2.657+- 0.044  2.128+- 0.033  2.349+- 0.036  2.332+- 0.021
err   0.909+- 0.034  0.952+-  0.03  0.955+-  0.03  0.985+- 0.019
fit_r2017d_v1_a
val   2.852+- 0.044  2.287+- 0.037  2.512+- 0.037  2.512+- 0.023
err   0.925+- 0.042  1.019+- 0.032  0.925+- 0.031  1.018+-  0.02
fit_r2017e_v1_a
val    3.04+- 0.042  2.448+- 0.034  2.836+- 0.036  2.742+- 0.021
err   0.872+- 0.034  0.856+- 0.028    0.9+-  0.03  0.932+- 0.017
fit_r2017f_v1_a
val    3.18+- 0.048  2.541+- 0.037  2.868+- 0.039  2.817+- 0.024
err   1.036+- 0.037  1.038+- 0.033  1.038+- 0.035  1.092+- 0.021
fit_r2017g_v1_a
val   0.035+- 0.155 -2.666+- 0.116 -4.415+- 0.099 -2.728+- 0.084
err    3.17+- 0.141  3.098+- 0.102   2.42+- 0.087  3.351+- 0.062
fit_r2017h_v1_a
val   0.116+- 0.079 -2.594+- 0.057 -4.639+- 0.055 -2.684+- 0.061
err   1.705+- 0.077  1.531+- 0.052   1.43+- 0.041  2.384+- 0.041
fit_r2017ae_ar1
val   2.569+- 0.032  1.989+- 0.027  2.265+- 0.029  2.238+- 0.017
err   0.711+- 0.027  0.738+- 0.023  0.761+- 0.026  0.795+- 0.015
fit_r2017af_ar1
val    2.68+- 0.032  2.064+- 0.027   2.36+- 0.029  2.328+- 0.018
err   0.694+- 0.025  0.748+- 0.023   0.76+- 0.025  0.804+- 0.014




  */
}

void all_2d() {
  TCanvas* c1 = new TCanvas("c1", "c1", 700, 400);
  //  c1->Divide(2,2);

  _file0->cd();
  const_term_2d();
  _file1->cd();
  const_term_2d();
  _file2->cd();
  const_term_2d();
  _file3->cd();
  const_term_2d();
  _file4->cd();
  const_term_2d();
  _file5->cd();
  const_term_2d();
  _file6->cd();
  const_term_2d();
  _file7->cd();
  const_term_2d();
  _file8->cd();
  const_term_2d();
  _file9->cd();
  const_term_2d();
  _file10->cd();
  const_term_2d();
  _file11->cd();
  const_term_2d();
  _file12->cd();
  const_term_2d();
  _file13->cd();
  const_term_2d();
}

void const_term_1d4x(const char* varname = "const_eta_phi") {
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(111);
  gStyle->SetOptStat(1110);
  gStyle->SetOptLogy(1);
  gStyle->SetPadTopMargin(0.01);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.01);
  gStyle->SetStatY(.99);  //89);
  gStyle->SetStatX(.99);  //94);
  gStyle->SetStatH(.22);
  gStyle->SetStatW(.27);

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.075);
  latex.SetTextFont(42);
  latex.SetTextAlign(31);  // align right

  TCanvas* c1 = new TCanvas("c1", "c1", 700, 400);
  c1->Divide(2, 2);

  //  for (int ixj=0; ixj<14; ixj++) {
  //     switch(ixj) {
  //     case 0 : _file0->cd(); break;
  //     case 1 : _file1->cd(); break;
  //     case 2 : _file2->cd(); break;
  //     case 3 : _file3->cd(); break;
  //     case 4 : _file4->cd(); break;
  //     case 5 : _file5->cd(); break;
  //     case 6 : _file6->cd(); break;
  //     case 7 : _file7->cd(); break;
  //     case 8 : _file8->cd(); break;
  //     case 9 : _file9->cd(); break;
  //     case 10 : _file10->cd(); break;
  //     case 11 : _file11->cd(); break;
  //     case 12 : _file12->cd(); break;
  //     case 13 : _file13->cd(); break;
  //     default : _file0->cd(); break;
  //     }

  TH2F* hist2d = (TH2F*)gDirectory->Get(varname);
  TH1F* histx[4];
  histx[0] = new TH1F("histx0", "Correction factor - 1 (R0)", 120, -0.11, 0.29);     //.9, 1.3);
  histx[1] = new TH1F("histx1", "Correction factor - 1 (R#pm1)", 120, -0.11, 0.29);  //.9, 1.3);
  histx[2] = new TH1F("histx2", "Correction factor - 1 (R#pm2)", 120, -0.11, 0.29);  //.9, 1.3);
  histx[3] = new TH1F("histx3", "Correction factor - 1 (All)", 120, -0.11, 0.29);    //.9, 1.3);

  char name[100] = gDirectory->GetName();
  char namex[20];
  char namey[50];
  strncpy(namex, name, 15);
  char* pchy = strchr(namex, '_');
  int len = pchy - namex;
  strncpy(namey, namex, len);

  cout << namex << " " << namey << endl;
  float value[2][4] = {0};
  float error[2][4] = {0};

  for (int ij = 0; ij < hist2d->GetNbinsX(); ij++) {
    int ieta = getieta(ij);
    for (int jk = 0; jk < hist2d->GetNbinsY(); jk++) {
      double xx = hist2d->GetBinContent(ij + 1, jk + 1) - 1.0;
      if (xx > -0.5) {
        if (abs(ieta) <= 4) {
          histx[0]->Fill(xx);
        } else if (abs(ieta) <= 10) {
          histx[1]->Fill(xx);
        } else if (abs(ieta) <= 15) {
          histx[2]->Fill(xx);
        }
        histx[3]->Fill(xx);
      }
    }
  }
  for (int ij = 0; ij < 4; ij++) {
    c1->cd(ij + 1);
    histx[ij]->GetXaxis()->SetTitle(histx[ij]->GetTitle());
    histx[ij]->GetXaxis()->SetLabelOffset(.001);
    histx[ij]->GetXaxis()->SetLabelSize(0.0645);
    histx[ij]->GetXaxis()->SetTitleSize(0.065);
    histx[ij]->GetXaxis()->SetTitleOffset(0.99);

    histx[ij]->GetYaxis()->SetTitle("Entries/0.005");
    histx[ij]->GetYaxis()->SetLabelOffset(.001);
    histx[ij]->GetYaxis()->SetLabelSize(0.0645);
    histx[ij]->GetYaxis()->SetTitleSize(0.065);
    histx[ij]->GetYaxis()->SetTitleOffset(0.85);

    histx[ij]->SetLineWidth(1);
    TFitResultPtr ptr = histx[ij]->Fit("gaus", "QS");
    Int_t fitStatus = ptr;
    if (fitStatus == 0) {
      value[0][ij] = int(100000 * ptr->Parameter(1)) / 1000.;
      value[1][ij] = int(100000 * ptr->Parameter(2)) / 1000.;
      error[0][ij] = int(100000 * ptr->ParError(1)) / 1000.;
      error[1][ij] = int(100000 * ptr->ParError(2)) / 1000.;

      //      cout<<ij<<" "<<int(100000*ptr->Parameter(1))/1000.<<"+-"<< int(100000*ptr->ParError(1))/1000.<<" "<<int(100000*ptr->Parameter(2))/1000.<<"+-"<< int(100000*ptr->ParError(2))/1000.<<endl;
    }
  }
  //    latex.DrawLatex(0.85, 0.45, namex);

  sprintf(namey, "const_term_1d4x_%s.png", namex);
  c1->SaveAs(namey);

  cout << "val ";
  for (int ij = 0; ij < 4; ij++) {
    cout << " " << setw(6) << value[0][ij] << "+-" << setw(6) << error[0][ij];
  }
  cout << endl;
  cout << "err ";
  for (int ij = 0; ij < 4; ij++) {
    cout << " " << setw(6) << value[1][ij] << "+-" << setw(6) << error[1][ij];
  }
  cout << endl;
  for (int ij = 0; ij < 4; ij++) {
    if (histx[ij]) {
      delete histx[ij];
      histx[ij] = 0;
    }
  }
  //  }
}

void const_term_1dx(const char* varname = "const_eta_phi") {
  ofstream file_out("test_output.log");
  gStyle->SetOptFit(101);
  gStyle->SetOptStat(0);
  gStyle->SetOptLogy(1);
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadBottomMargin(0.11);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadRightMargin(0.03);
  gStyle->SetStatY(.93);  //89);
  gStyle->SetStatX(.98);  //94);
  gStyle->SetStatH(.16);
  gStyle->SetStatW(.19);

  TH2F* hist2d[3];

  hist2d[0] = (TH2F*)gDirectory->Get(varname);
  hist2d[1] = (TH2F*)gDirectory->Get("const_eta_phisum");
  TH1F* histx[3];
  histx[0] = new TH1F("histx0", "const_eta_phi", 120, .7, 1.3);
  histx[1] = new TH1F("histx1", "const_eta_phisum", 120, .7, 1.3);
  histx[2] = new TH1F("histx2", "Correction term", 120, -.2, .2);
  hist2d[2] = new TH2F("hist2d2", "hist2d2", 120, .9, 1.1, 120, .9, 1.1);
  for (int ij = 0; ij < hist2d[0]->GetNbinsX(); ij++) {
    //    if (hist2d[0]->GetBinCenter(ij+1)==0) continue;
    for (int jk = 0; jk < hist2d[0]->GetNbinsY(); jk++) {
      double xx = hist2d[0]->GetBinContent(ij + 1, jk + 1);
      if (xx > 0.5) {
        histx[0]->Fill(xx);
      }
      if (ij != 15)
        file_out << hist2d[0]->GetXaxis()->GetBinCenter(ij + 1) << "\t" << jk + 1 << "\t " << xx << endl;

      double yy = hist2d[1]->GetBinContent(ij + 1, jk + 1);
      if (yy > 0.5) {
        histx[1]->Fill(yy);
      }
      if (xx > .5 && yy > .5) {
        histx[2]->Fill(yy - xx);
        hist2d[2]->Fill(xx, yy);
      }
    }
  }
  TCanvas* c1 = new TCanvas("c1", "c1", 500, 500);
  //  c1->Divide(2,2);
  /*  c1->cd(1);  histx[0]->Fit("gaus");
      c1->cd(2);  histx[1]->Fit("gaus");
      c1->cd(3); histx[2]->Fit("gaus");
  */

  histx[0]->GetXaxis()->SetTitle("Correction factor");
  histx[0]->GetXaxis()->SetLabelOffset(.001);
  histx[0]->GetXaxis()->SetLabelSize(0.0645);
  histx[0]->GetXaxis()->SetTitleSize(0.065);
  histx[0]->GetXaxis()->SetTitleOffset(0.75);

  histx[0]->GetYaxis()->SetTitle("Entries/0.005");
  histx[0]->GetYaxis()->SetLabelOffset(.001);
  histx[0]->GetYaxis()->SetLabelSize(0.0645);
  histx[0]->GetYaxis()->SetTitleSize(0.065);
  histx[0]->GetYaxis()->SetTitleOffset(1.05);

  histx[0]->SetLineWidth(1);
  histx[0]->Fit("gaus");
  //  c1->cd(4);  gPad->SetLogy(0); hist2d[2]->Draw("colz");
  //  cmsPrel(.25, .45, .85, .62, 0.04, 35);
  file_out.close();
}

void const_term_2dx(const char* varname = "const_eta_phi") {
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetOptLogy(0);

  gStyle->SetPalette(1, 0);
  gStyle->SetPadTopMargin(0.12);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadLeftMargin(0.08);
  gStyle->SetPadRightMargin(0.15);

  TH2F* hist2d = (TH2F*)gDirectory->Get(varname);

  hist2d->SetMaximum(1.5);
  hist2d->SetMinimum(0.7);
  hist2d->GetXaxis()->SetTitle("i#eta");
  hist2d->GetYaxis()->SetTitle("i#phi");

  hist2d->Draw("colz");

  cmsPrel2(.75, .5, .15, .92, 0.025, 35);
}

double hothreshs[nhothresh + 1] = {0.15, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20., 50, 100., 10000.};
void def_setting() {
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadLeftMargin(0.08);
  gStyle->SetPadRightMargin(0.16);
  gStyle->SetOptStat(0);
  //  gStyle->SetPadGridX(1);
  //  gStyle->SetPadGridY(1);
  //  gStyle->SetGridStyle(3);
  gStyle->SetGridWidth(1);
  gStyle->SetTitleFontSize(0.09);
  gStyle->SetTitleOffset(-0.09);
  gStyle->SetTitleBorderSize(1);
  gStyle->SetLabelSize(0.095, "XY");
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetNdivisions(404, "XYZ");

  gStyle->SetStatTextColor(1);
  gStyle->SetStatX(.99);
  gStyle->SetStatY(.99);
  gStyle->SetStatW(.3);
  gStyle->SetStatH(.2);

  gStyle->SetOptLogy(0);
}

void const_term(const char* varname = "const_eta_phi") {
  ofstream file_out("test_output.log");
  gStyle->SetOptFit(101);
  gStyle->SetOptStat(0);
  gStyle->SetOptLogy(1);
  gStyle->SetLabelSize(.065, "XY");

  TH2F* hist2d[3];

  hist2d[0] = (TH2F*)gDirectory->Get("const_eta_phi");
  hist2d[1] = (TH2F*)gDirectory->Get("const_eta_phisum");

  TH1F* histx[3];
  histx[0] = new TH1F("histx0", "const_eta_phi", 120, .7, 1.3);
  histx[1] = new TH1F("histx1", "const_eta_phisum", 120, .7, 1.3);
  histx[2] = new TH1F("histx2", "diff", 120, -.2, .2);
  hist2d[2] = new TH2F("hist2d2", "hist2d2", 120, .9, 1.1, 120, .9, 1.1);

  for (int ij = 0; ij < hist2d[0]->GetNbinsX(); ij++) {
    //    if (hist2d[0]->GetBinCenter(ij+1)==0) continue;
    for (int jk = 0; jk < hist2d[0]->GetNbinsY(); jk++) {
      double xx = hist2d[0]->GetBinContent(ij + 1, jk + 1);
      if (xx > 0.5) {
        histx[0]->Fill(xx);
      }
      if (ij != 15)
        file_out << hist2d[0]->GetXaxis()->GetBinCenter(ij + 1) << "\t" << jk + 1 << "\t " << xx << endl;

      double yy = hist2d[1]->GetBinContent(ij + 1, jk + 1);
      if (yy > 0.5) {
        histx[1]->Fill(yy);
      }
      if (xx > .5 && yy > .5) {
        histx[2]->Fill(yy - xx);
        hist2d[2]->Fill(xx, yy);
      }
    }
  }
  TCanvas* c1 = new TCanvas("c1", "c1", 700, 900);
  c1->Divide(2, 2);
  c1->cd(1);
  histx[0]->Fit("gaus");
  c1->cd(2);
  histx[1]->Fit("gaus");
  c1->cd(3);
  histx[2]->Fit("gaus");
  c1->cd(4);
  gPad->SetLogy(0);
  hist2d[2]->Draw("colz");
  file_out.close();
}

void const_term_2d(const char* varname = "const_eta_phi") {
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetPalette(1, 0);
  gStyle->SetPadTopMargin(0.09);
  gStyle->SetPadBottomMargin(0.11);
  gStyle->SetPadLeftMargin(0.11);
  gStyle->SetPadRightMargin(0.15);

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.065);
  latex.SetTextFont(42);
  latex.SetTextAlign(31);  // align right

  TH2F* hist2d = (TH2F*)gDirectory->Get(varname);

  hist2d->GetXaxis()->SetLabelOffset(.001);
  hist2d->GetXaxis()->SetLabelSize(0.064);
  hist2d->GetXaxis()->SetTitle("i#eta");
  hist2d->GetXaxis()->SetTitleSize(0.065);
  hist2d->GetXaxis()->SetTitleOffset(0.75);
  hist2d->GetXaxis()->SetTitleColor(1);
  hist2d->GetXaxis()->CenterTitle();

  hist2d->GetYaxis()->SetLabelOffset(.001);
  hist2d->GetYaxis()->SetLabelSize(0.064);
  hist2d->GetYaxis()->SetTitleSize(0.065);
  hist2d->GetYaxis()->SetTitle("i#phi");
  hist2d->GetYaxis()->SetTitleOffset(0.68);
  hist2d->GetYaxis()->CenterTitle();

  hist2d->GetZaxis()->SetLabelSize(0.045);

  //  hist2d->SetMaximum(1.2);
  hist2d->SetMinimum(0.7);
  hist2d->Draw("colz");  //"colz");
                         //   //c1->SaveAs("test.png");
                         //    cout <<"XX "<<endl;
                         //    TPaletteAxis* palette = new TPaletteAxis(15.7, 0.5, 18.5, 72.5, hist2d);
                         //    cout <<"XX "<<endl;
                         //    palette->SetLabelColor(1);
                         //    palette->SetLabelFont(22);
                         //    palette->SetLabelOffset(0.005);
                         //    palette->SetLabelSize(0.0436);
                         //    palette->SetTitleOffset(1);
                         //    palette->SetTitleSize(0.08);
                         //    palette->SetFillColor(100);
                         //    palette->SetFillStyle(1001);
                         //    hist2d->GetListOfFunctions()->Add(palette,"br");
                         //    hist2d->Draw("colz");
                         //    //  cmsPrel2(.75, .5, .15, .92, 0.03, 35);

  char name[100] = gDirectory->GetName();
  char namex[20];
  char namey[50];
  strncpy(namex, name, 15);
  cout << namex << endl;
  latex.DrawLatex(0.65, 0.92, namex);
  sprintf(namey, "statistics_2d_%s.png", namex);
  c1->SaveAs(namey);
}

void signal4(int ival = 0, int ndvy = 3) {
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.11);
  gStyle->SetPadBottomMargin(0.08);
  gStyle->SetPadLeftMargin(0.08);
  gStyle->SetPadRightMargin(0.20);
  TCanvas* c1 = new TCanvas("c1", "c1", 700, 900);

  char name[100];

  c1->Divide(3, ndvy);

  TH2F* histx[100];
  for (int ij = 0; ij < 3 * ndvy; ij++) {
    //    sprintf(name, "hoCalibc/%s_%i", varnam, ij);
    histx[ij] = (TH2F*)gDirectory->Get(name);
    switch (ival) {
      case 0:
        histx[ij] = (TH2F*)totalproj[ij]->Clone();
        break;
      case 1:
        histx[ij] = (TH2F*)total2proj[ij]->Clone();
        break;
      case 2:
        histx[ij] = (TH2F*)totalprojsig[ij]->Clone();
        break;
      case 3:
        histx[ij] = (TH2F*)total2projsig[ij]->Clone();
        break;
      case 4:
        histx[ij] = (TH2F*)totalhpd[ij]->Clone();
        break;
      case 5:
        histx[ij] = (TH2F*)total2hpd[ij]->Clone();
        break;
      case 6:
        histx[ij] = (TH2F*)totalhpdsig[ij]->Clone();
        break;
      case 7:
        histx[ij] = (TH2F*)total2hpdsig[ij]->Clone();
        break;
      default:
        break;
    }
    c1->cd(ij + 1);
    if (strstr(histx[ij]->GetName(), "sig")) {
      histx[ij]->SetMaximum(TMath::Min(3., histx[ij]->GetMaximum()));
    }
    histx[ij]->GetXaxis()->SetLabelSize(0.075);
    histx[ij]->GetYaxis()->SetLabelSize(0.075);
    histx[ij]->GetZaxis()->SetLabelSize(0.065);
    histx[ij]->GetXaxis()->SetNdivisions(404);
    histx[ij]->GetYaxis()->SetNdivisions(404);
    histx[ij]->Draw("colz");
  }
  c1->Update();
}

// plot_var("sigvsacc")
void plot_var(int icut) {
  const int nvar = 15;
  gStyle->SetTitleFontSize(0.075);
  gStyle->SetTitleBorderSize(1);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetPadBottomMargin(0.10);
  char title[100];
  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);
  c1->Divide(5, 3, 1.e-5, 1.e-5, 0);

  TH2F* fprofx[nvar];
  for (int ij = 0; ij < nvar; ij++) {
    c1->cd(ij + 1);
    sprintf(title, "hoCalibc/sigring_%i_%i", icut, ij);

    fprofx[ij] = (TH2F*)gDirectory->Get(title);
    fprofx[ij]->GetXaxis()->SetLabelOffset(-0.03);
    fprofx[ij]->GetXaxis()->SetLabelSize(0.085);
    fprofx[ij]->GetYaxis()->SetLabelSize(0.075);
    fprofx[ij]->GetZaxis()->SetLabelSize(0.065);
    fprofx[ij]->GetYaxis()->SetNdivisions(404);
    fprofx[ij]->GetXaxis()->SetNdivisions(404);
    fprofx[ij]->SetLineColor(2);
    fprofx[ij]->SetLineWidth(2);

    fprofx[ij]->Draw("colz");
    fprofx[ij]->ProfileX()->Draw("same");

    //     if (ij==7) {
    //       TPaveStats *ptst = new TPaveStats(0.5,0.5,0.7,0.6,"brNDC");
    //       ptst->SetFillColor(10);
    //       TText* text = ptst->AddText(var);
    //       text->SetTextSize(0.092);
    //       ptst->SetBorderSize(1);
    //       //   ptst->AddText(name);
    //       ptst->Draw();
    //     }
  }
  c1->Update();
}

// plot_var("sigvsacc")
void plot_varprof(int icut) {
  const int nvar = 15;
  gStyle->SetTitleFontSize(0.075);
  gStyle->SetTitleBorderSize(1);
  gStyle->SetPadTopMargin(0.12);
  gStyle->SetPadBottomMargin(0.10);
  char title[100];
  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);
  c1->Divide(5, 3, 1.e-5, 1.e-5, 0);

  TProfile* fprofx[nvar];
  for (int ij = 0; ij < nvar; ij++) {
    c1->cd(ij + 1);
    sprintf(title, "hoCalibc/sigring_%i_%i", icut, ij);

    fprofx[ij] = (TProfile*)(((TH2F*)gDirectory->Get(title))->ProfileX());
    fprofx[ij]->GetXaxis()->SetLabelOffset(-0.03);
    fprofx[ij]->GetXaxis()->SetLabelSize(0.085);
    fprofx[ij]->GetYaxis()->SetLabelSize(0.075);
    fprofx[ij]->GetYaxis()->SetNdivisions(404);
    fprofx[ij]->GetXaxis()->SetNdivisions(404);
    fprofx[ij]->SetLineColor(2);
    fprofx[ij]->SetLineWidth(2);

    fprofx[ij]->Draw();

    //     if (ij==7) {
    //       TPaveStats *ptst = new TPaveStats(0.5,0.5,0.7,0.6,"brNDC");
    //       ptst->SetFillColor(10);
    //       TText* text = ptst->AddText(var);
    //       text->SetTextSize(0.092);
    //       ptst->SetBorderSize(1);
    //       //   ptst->AddText(name);
    //       ptst->Draw();
    //     }
  }
  c1->Update();
}

void plot_var_tray(int id = 0, int icut = 0) {
  //plot_var_tray(0,1)

  char title[100];
  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);
  c1->Divide(5, 6, 1.e-5, 1.e-5, 0);

  TH2F* fprofx[30];
  int itag = 0;
  for (int jk = 0; jk < 6; jk++) {
    for (int ij = 0; ij < 5; ij++) {
      c1->cd(itag + 1);
      sprintf(title, "hoCalibc/sigtray_%i_%i_%i", icut, 5 * id + ij, jk);

      fprofx[itag] = (TH2F*)gDirectory->Get(title);
      fprofx[itag]->GetXaxis()->SetLabelOffset(-.001);
      fprofx[itag]->GetXaxis()->SetLabelSize(0.085);
      fprofx[itag]->GetYaxis()->SetLabelSize(0.085);
      fprofx[itag]->GetZaxis()->SetLabelSize(0.065);
      fprofx[itag]->GetYaxis()->SetNdivisions(404);
      fprofx[itag]->GetXaxis()->SetNdivisions(404);
      fprofx[itag]->SetLineColor(2);
      fprofx[itag]->SetLineWidth(2);

      fprofx[itag]->Draw("colz");
      fprofx[itag]->ProfileX()->Draw("same");
      //       if (itag==7) {
      // 	TPaveStats *ptst = new TPaveStats(0.5,0.5,0.7,0.6,"brNDC");
      // 	ptst->SetFillColor(10);
      // 	TText* text = ptst->AddText(var);
      // 	text->SetTextSize(0.092);
      // 	ptst->SetBorderSize(1);
      // 	//   ptst->AddText(name);
      // 	ptst->Draw();
      //       }

      itag++;
    }
  }
  c1->Update();
}

void plot_var_trayprof(int id = 0, int icut = 0) {
  //plot_var_tray(0,1)

  char title[100];
  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);
  c1->Divide(5, 6, 1.e-5, 1.e-5, 0);

  TProfile* fprofx[30];
  int itag = 0;
  for (int jk = 0; jk < 6; jk++) {
    for (int ij = 0; ij < 5; ij++) {
      c1->cd(itag + 1);
      sprintf(title, "hoCalibc/sigtray_%i_%i_%i", icut, 5 * id + ij, jk);

      fprofx[itag] = (TProfile*)(((TH2F*)gDirectory->Get(title))->ProfileX());
      fprofx[itag]->GetXaxis()->SetLabelOffset(-.001);
      fprofx[itag]->GetXaxis()->SetLabelSize(0.085);
      fprofx[itag]->GetYaxis()->SetLabelSize(0.085);
      fprofx[itag]->GetYaxis()->SetNdivisions(404);
      fprofx[itag]->GetXaxis()->SetNdivisions(404);
      fprofx[itag]->SetLineColor(2);
      fprofx[itag]->SetLineWidth(2);

      fprofx[itag]->Draw();
      //       if (itag==7) {
      // 	TPaveStats *ptst = new TPaveStats(0.5,0.5,0.7,0.6,"brNDC");
      // 	ptst->SetFillColor(10);
      // 	TText* text = ptst->AddText(var);
      // 	text->SetTextSize(0.092);
      // 	ptst->SetBorderSize(1);
      // 	//   ptst->AddText(name);
      // 	ptst->Draw();
      //       }

      itag++;
    }
  }
  c1->Update();
}

void plot_var_eta(int id = 0, int icut = 0) {
  //plot_var_eta(0,1)

  const int netamx = 30;

  char title[100];
  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);
  c1->Divide(5, 6, 1.e-5, 1.e-5, 0);

  TH2F* fprofx[netamx];
  for (int ij = 0; ij < netamx; ij++) {
    c1->cd(ij + 1);
    sprintf(title, "hoCalibc/sigeta_%i_%i_%i", icut, id, ij);
    fprofx[ij] = (TH2F*)gDirectory->Get(title);
    fprofx[ij]->GetXaxis()->SetLabelOffset(-0.001);
    fprofx[ij]->GetXaxis()->SetLabelSize(0.085);
    fprofx[ij]->GetYaxis()->SetLabelSize(0.085);
    fprofx[ij]->GetZaxis()->SetLabelSize(0.065);
    fprofx[ij]->GetYaxis()->SetNdivisions(404);
    fprofx[ij]->GetXaxis()->SetNdivisions(404);
    fprofx[ij]->SetLineColor(2);
    fprofx[ij]->SetLineWidth(2);

    fprofx[ij]->Draw("colz");
    fprofx[ij]->ProfileX()->Draw("same");

    //     if (ij==7) {
    //       TPaveStats *ptst = new TPaveStats(0.5,0.5,0.7,0.6,"brNDC");
    //       ptst->SetFillColor(10);
    //       TText* text = ptst->AddText(var);
    //       text->SetTextSize(0.092);
    //       ptst->SetBorderSize(1);
    //       //   ptst->AddText(name);
    //       ptst->Draw();
    //     }
  }
  c1->Update();
}

void plot_var_etaprof(int id = 0, int icut = 0) {
  //plot_var_eta(0,1)

  const int netamx = 30;

  char title[100];
  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);
  c1->Divide(5, 6, 1.e-5, 1.e-5, 0);

  TProfile* fprofx[netamx];
  for (int ij = 0; ij < netamx; ij++) {
    c1->cd(ij + 1);
    sprintf(title, "hoCalibc/sigeta_%i_%i_%i", icut, id, ij);
    fprofx[ij] = (TProfile*)(((TH2F*)gDirectory->Get(title))->ProfileX());
    fprofx[ij]->GetXaxis()->SetLabelOffset(-0.001);
    fprofx[ij]->GetXaxis()->SetLabelSize(0.085);
    fprofx[ij]->GetYaxis()->SetLabelSize(0.085);
    fprofx[ij]->GetYaxis()->SetNdivisions(404);
    fprofx[ij]->GetXaxis()->SetNdivisions(404);
    fprofx[ij]->SetLineColor(2);
    fprofx[ij]->SetLineWidth(2);

    fprofx[ij]->Draw();
  }
  c1->Update();
}

void plotyx() {
  //  //  sprintf (outfile,"%s.ps",outfilx);
  //  TPostScript ps("test.ps",111);
  //  ps.Range(20,28);
  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);

  for (int kl = 0; kl < ncut; kl++) {
    //    ps.NewPage();
    plot_var(kl);
  }
  //  ps.Close();
}

void plotallx() {
  //   //  gDirectory->DeleteAll();
  //   gStyle->Reset();
  gStyle->SetTitleFontSize(0.075);
  gStyle->SetTitleBorderSize(1);
  gStyle->SetPadTopMargin(0.12);
  gStyle->SetPadBottomMargin(0.10);
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  gStyle->SetGridStyle(3);
  gStyle->SetOptStat(0);
  gStyle->SetTitleBorderSize(1);

  TCanvas* c1 = new TCanvas("c1", "runfile", 700., 900.);
  int ips = 111;
  //  //  sprintf (outfile,"%s.ps",outfilx);
  TPostScript ps("test_2016e.ps", ips);
  ps.Range(20, 28);
  bool m_select_plot = true;
  //   cout<<"1xx "<<endl;
  if (m_select_plot) {
    for (int kl = 0; kl < ncut; kl++) {
      ps.NewPage();
      plot_var(kl);
      ps.NewPage();
      plot_varprof(kl);
    }

    gStyle->SetTitleFontSize(0.095);
    for (int ix = 0; ix < 3; ix++) {
      for (int iy = 0; iy < ncut; iy++) {
        ps.NewPage();
        plot_var_eta(ix, iy);
        ps.NewPage();
        plot_var_etaprof(ix, iy);
      }
    }

    for (int ix = 0; ix < 3; ix++) {
      for (int iy = 0; iy < ncut; iy++) {
        ps.NewPage();
        plot_var_tray(ix, iy);
        ps.NewPage();
        plot_var_trayprof(ix, iy);
      }
    }
  }

  char name[100];
  totalmuon = (TH2F*)gDirectory->Get("hoCalibc/totalmuon");
  total2muon = (TH2F*)gDirectory->Get("hoCalibc/total2muon");

  for (int ij = 0; ij < 9; ij++) {
    sprintf(name, "hoCalibc/totalproj_%i", ij);
    totalproj[ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/totalprojsig_%i", ij);
    totalprojsig[ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/total2proj_%i", ij);
    total2proj[ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/total2projsig_%i", ij);
    total2projsig[ij] = (TH2F*)gDirectory->Get(name);

    total2projsig[ij]->Divide(total2proj[ij]);
    totalprojsig[ij]->Divide(totalproj[ij]);

    total2proj[ij]->Divide(total2muon);
    totalproj[ij]->Divide(totalmuon);
  }

  for (int ij = 0; ij < 18; ij++) {
    sprintf(name, "hoCalibc/totalhpd_%i", ij);
    totalhpd[ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/totalhpdsig_%i", ij);
    totalhpdsig[ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/total2hpd_%i", ij);
    total2hpd[ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/total2hpdsig_%i", ij);
    total2hpdsig[ij] = (TH2F*)gDirectory->Get(name);

    total2hpdsig[ij]->Divide(total2hpd[ij]);
    totalhpdsig[ij]->Divide(totalhpd[ij]);

    total2hpd[ij]->Divide(total2muon);
    totalhpd[ij]->Divide(totalmuon);
  }
  for (int ij = 0; ij < 8; ij++) {
    ps.NewPage();
    signal4(ij, (ij <= 3) ? 3 : 6);
  }
  //   cout<<"8xx "<<endl;
  // //   ps.NewPage(); signal4("hoCalibc/totalproj",3);
  // //   ps.NewPage(); signal4("hoCalibc/total2proj",3);
  // //   ps.NewPage(); signal4("hoCalibc/totalprojsig",3);
  // //   ps.NewPage(); signal4("hoCalibc/total2projsig",3);

  // //   ps.NewPage(); signal4("hoCalibc/totalhpd",6);
  // //   ps.NewPage(); signal4("hoCalibc/total2hpd",6);
  // //   ps.NewPage(); signal4("hoCalibc/totalhpdsig",6);
  // //   ps.NewPage(); signal4("hoCalibc/total2hpdsig",6);

  //  ps.Close();
}

void signal1() {
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(4, 2);
  c1->cd(1);
  totalmuon->Draw("colz");
  c1->cd(2);
  totalproj_3->Draw("colz");
  c1->cd(3);
  totalproj_4->Draw("colz");
  c1->cd(4);
  totalproj_5->Draw("colz");
  c1->cd(5);
  total2muon->Draw("colz");
  c1->cd(6);
  total2proj_3->Draw("colz");
  c1->cd(7);
  total2proj_4->Draw("colz");
  c1->cd(8);
  total2proj_5->Draw("colz");
}

void signal2() {
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(4, 2);
  c1->cd(1);
  totalmuon->Draw("colz");
  c1->cd(2);
  totalproj_3->Draw("colz");
  c1->cd(3);
  totalproj_4->Draw("colz");
  c1->cd(4);
  totalproj_5->Draw("colz");

  c1->cd(6);
  totalprojsig_3->Draw("colz");
  c1->cd(7);
  totalprojsig_4->Draw("colz");
  c1->cd(8);
  totalprojsig_5->Draw("colz");
}
void signal3() {
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(4, 2);
  c1->cd(1);
  total2muon->Draw("colz");
  c1->cd(2);
  total2proj_3->Draw("colz");
  c1->cd(3);
  total2proj_4->Draw("colz");
  c1->cd(4);
  total2proj_5->Draw("colz");

  c1->cd(6);
  total2projsig_3->Draw("colz");
  c1->cd(7);
  total2projsig_4->Draw("colz");
  c1->cd(8);
  total2projsig_5->Draw("colz");
}

//signal4("totalproj",3);
//signal4("total2proj",3);
//signal4("totalprojsig",3);
//signal4("total2projsig",3);

//signal4("totalhpd",6);
//signal4("total2hpd",6);
//signal4("totalhpdsig",6);
//signal4("total2hpdsig",6);

// void signal4(const char* varnam="total2proj", int ndvy=3) {
//   gStyle->SetOptStat(0);

//   TCanvas* c1 = new TCanvas("c1", "c1", 1200., 600.);

//   char name[100];
//   TH2F* histx[100];
//   c1->Divide(ndvy, 3);
//   for (int ij=0; ij <3*ndvy; ij++) {
//     sprintf(name, "%s_%i", varnam, ij);
//     histx[ij] = (TH2F*)gDirectory->Get(name);
//     c1->cd(ij+1);
//     histx[ij]->Draw("colz");
//   }
// }

void testplotx2(int thmn = 2, int thmx = 6) {
  gStyle->SetPadTopMargin(0.11);
  gStyle->SetPadBottomMargin(0.04);
  gStyle->SetPadLeftMargin(0.06);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle();
  gStyle->SetPadGridX(1);
  gStyle->SetPadGridY(1);
  gStyle->SetGridStyle(3);
  gStyle->SetGridWidth(1);
  gStyle->SetTitleColor(10);
  gStyle->SetTitleFontSize(0.09);
  gStyle->SetTitleOffset(-0.09);
  gStyle->SetTitleBorderSize(1);
  gStyle->SetLabelSize(0.095, "XY");

  //  const int nfile=3;
  //  char* indexx[nfile]={"1_27", "28_122", "123_181"};

  const int nringmx = 5;

  const int routmx = 36;
  const int rout12mx = 24;

  const int rbxmx = 12;   // HO readout box in Ring 0
  const int rbx12mx = 6;  //HO readout box in Ring+-1/2

  const int nprojmx = 4;
  const int nseltype = 4;  //Different crit for muon selection
  const int nadmx = 18;
  const int shapemx = 10;  //checking shape

  char* projname[nprojmx] = {"totalproj", "totalprojsig", "totalhpd", "totalhpdsig"};

  //  TH2F* total2muon;

  TH2F* totalmuon[nseltype];
  TH2F* totalproj[nseltype][nprojmx][nadmx];

  const int nprojtype = 5;  // Varities due to types of muon projection
  TH2F* histent[nprojtype + 3];
  TH2F* histen[nprojtype + 3];
  TH2F* histen2[nprojtype + 3];
  TH2F* histerr[nprojtype];

  const int nhothresh = 10;
  TH2F* h_allmucorrel[nhothresh];
  TH2F* hnorm_allmucorrel[nhothresh];

  TH1F* endigicnt[nprojtype + 1][nhothresh];
  TH1F* endigisig[nprojtype + 1][nhothresh];

  TH2F* rmdigicnt[nprojtype + 1][nhothresh];
  TH2F* rmdigisig[nprojtype + 1][nhothresh];

  TH2F* ringdigicnt[nprojtype + 1][nhothresh];
  TH2F* ringdigisig[nprojtype + 1][nhothresh];

  TH2F* inddigicnt[nprojtype + 1][nhothresh];
  TH2F* inddigisig[nprojtype + 1][nhothresh];

  TH2F* indrecocnt[nprojtype + 1][nhothresh];
  TH2F* indrecosig[nprojtype + 1][nhothresh];

  TH1F* rout_mult[nhothresh][nringmx][routmx + 1];
  TH1F* rbx_mult[nhothresh][nringmx][rbxmx + 1];

  TH1F* rout_ind_energy[nringmx][routmx + 1];

  TH2F* h_correlht[nhothresh];
  TH2F* h_correlsig[nringmx][nhothresh];
  TH2F* h_rmoccu[nhothresh];
  TH2F* h_rmcorrel[nhothresh];

  TH2F* h_allcorrelsig[nhothresh];

  TH2F* rbx_shape[shapemx][nringmx][routmx + 1];

  char name[100];

  int ips = 111;
  TPostScript ps("allmuho.ps", ips);
  ps.Range(20, 28);

  TFile* fx = new TFile("histall_apr14b_cosmic_csa14_cosmic.root", "read");
  //  TFile* fx = new TFile("hist_apr14c_cosmic_csa14_cosmic.root", "read");
  //  TFile* fx = new TFile("histall_cosmic_csa14_cosmic_set.root", "read");
  //  TFile* fx = new TFile("apr14/apr14b/hist_cosmic_csa14_cosmic_set52.root", "read");
  for (int isel = 0; isel < nseltype; isel++) {
    sprintf(name, "hoCalibc/totalmuon_%i", isel);
    totalmuon[isel] = (TH2F*)fx->Get(name);
  }
  //  total2muon = (TH2F*) fx->Get("hoCalibc/total2muon");

  TCanvas* c0 = new TCanvas("c0", "mean rms", 600, 800);
  c0->Divide(2, 2);

  TCanvas* c3 = new TCanvas("c3", "mean rms", 600, 800);

  TCanvas* c1 = new TCanvas("c1", "c1", 600, 900);
  c1->Divide(3, 3);

  TCanvas* c2 = new TCanvas("c2", "c2", 600, 900);
  c2->Divide(3, 6);

  TCanvas* c4x = new TCanvas("c4x", "c4x", 600., 800.);
  c4x->Divide(2, 6);

  TCanvas* c4 = new TCanvas("c4", "c4", 600., 800.);
  c4->Divide(2, 6);

  TCanvas* c5 = new TCanvas("c5", "c5", 600., 800.);
  c5->Divide(6, 6);

  TCanvas* c6 = new TCanvas("c6", "c6", 600., 800.);
  c6->Divide(2, 5);

  TH2F* h2d_nhocapid[nprojtype];
  TH2F* h2d_hocapidsig[nprojtype];
  TH2F* h2d_hocapidsigwo[nprojtype];

  for (int ij = 0; ij < nprojtype; ij++) {
    sprintf(name, "hoCalibc/nhocapid_%i", ij);
    h2d_nhocapid[ij] = (TH2F*)fx->Get(name);

    sprintf(name, "hoCalibc/hocapidsig_%i", ij);
    h2d_hocapidsig[ij] = (TH2F*)fx->Get(name);

    sprintf(name, "hoCalibc/hocapidsigwo_%i", ij);
    h2d_hocapidsigwo[ij] = (TH2F*)fx->Get(name);

    h2d_hocapidsig[ij]->Divide(h2d_nhocapid[ij]);
    h2d_hocapidsigwo[ij]->Divide(h2d_nhocapid[ij]);

    ps.NewPage();
    c3->cd();
    h2d_hocapidsig[ij]->Draw("colz");
    c3->Update();

    ps.NewPage();
    c3->cd();
    h2d_hocapidsigwo[ij]->Draw("colz");
    c3->Update();
  }

  for (int ij = 0; ij < nprojtype; ij++) {
    sprintf(name, "hoCalibc/hbentry_d%i", ij + 3);
    cout << "ij " << name << endl;
    histent[ij] = (TH2F*)fx->Get(name);
    //    histent[ij]->Draw();
    sprintf(name, "hoCalibc/hbsig_d%i", ij + 3);
    histen[ij] = (TH2F*)fx->Get(name);
    sprintf(name, "hoCalibc/hbsig2_d%i", ij + 3);
    histen2[ij] = (TH2F*)fx->Get(name);

    int nentry = TMath::Max(1., histent[ij]->GetBinContent(0, 0));
    cout << "ij " << nentry << endl;

    histen[ij]->Divide(histent[ij]);
    histen2[ij]->Divide(histent[ij]);

    histent[ij]->Scale(1. / nentry);

    int xbin = histen[ij]->GetNbinsX();
    int ybin = histen[ij]->GetNbinsY();
    sprintf(name, "histerr_%i", ij);  //, indexx[ij]);
    histerr[ij] = new TH2F(name,
                           name,
                           xbin,
                           histent[ij]->GetXaxis()->GetXmin(),
                           histent[ij]->GetXaxis()->GetXmax(),
                           ybin,
                           histent[ij]->GetYaxis()->GetXmin(),
                           histent[ij]->GetYaxis()->GetXmax());

    for (int jk = 1; jk <= xbin; jk++) {
      for (int kl = 1; kl <= ybin; kl++) {
        double err = sqrt(histen2[ij]->GetBinContent(jk, kl) -
                          histen[ij]->GetBinContent(jk, kl) * histen[ij]->GetBinContent(jk, kl));

        histerr[ij]->Fill(histent[ij]->GetXaxis()->GetBinCenter(jk), histent[ij]->GetYaxis()->GetBinCenter(kl), err);
      }
    }

    ps.NewPage();
    cout << "ij11 " << ij << " " << histent[ij]->GetBinContent(22, 22) << endl;
    c0->cd(1); /*histent[ij]->GetXaxis()->SetRangeUser(-15.49,15.49);*/
    histent[ij]->Draw("colz");
    cout << "ij12 " << ij << " " << histen[ij]->GetBinContent(22, 22) << endl;
    c0->cd(2);
    histen[ij]->GetXaxis()->SetRangeUser(-15.49, 15.49);
    histen[ij]->Draw("colz");
    cout << "ij13 " << ij << endl;
    c0->cd(3);
    histen2[ij]->GetXaxis()->SetRangeUser(-15.49, 15.49);
    histen2[ij]->Draw("colz");
    cout << "ij14 " << ij << endl;
    c0->cd(4);
    histerr[ij]->GetXaxis()->SetRangeUser(-15.49, 15.49);
    histerr[ij]->Draw("colz");
    cout << "ij15 " << ij << endl;
    c0->Update();
    cout << "ij16 " << ij << endl;
  }

  for (int jk = 0; jk < nringmx; jk++) {
    for (int kl = 0; kl <= rbxmx; kl++) {
      if (jk != 2 && kl > rbx12mx)
        continue;
      ps.NewPage();
      for (int lm = 0; lm < shapemx; lm++) {
        if (kl == 0) {
          sprintf(name, "hoCalibc/rbx_shape_%i_r%i_allrbx", lm, jk);
        } else {
          sprintf(name, "hoCalibc/rbx_shape_%i_r%i_%i", lm, jk, kl);
        }
        rbx_shape[lm][jk][kl] = (TH2F*)fx->Get(name);
        c6->cd(int(lm / 5) + 2 * (lm % 5) + 1);

        rbx_shape[lm][jk][kl]->Draw("colz");
      }
      c6->Update();
    }
  }

  //Correlations
  for (int ij = thmn; ij < thmx; ij++) {
    sprintf(name, "hoCalibc/hocorrelht_%i", ij);
    h_correlht[ij] = (TH2F*)fx->Get(name);

    h_correlht[ij]->Scale(1. / nentry);

    sprintf(name, "hoCalibc/hormoccu_%i", ij);
    h_rmoccu[ij] = (TH2F*)fx->Get(name);
    h_rmoccu[ij]->Scale(1. / nentry);

    sprintf(name, "hoCalibc/hormcorrel_%i", ij);
    h_rmcorrel[ij] = (TH2F*)fx->Get(name);
    h_rmcorrel[ij]->Scale(1. / nentry);

    for (int jk = 0; jk < nringmx; jk++) {
      sprintf(name, "hoCalibc/hocorrelsig_%i_%i", jk, ij);
      h_correlsig[jk][ij] = (TH2F*)fx->Get(name);
      h_correlsig[jk][ij]->Scale(1. / nentry);
    }

    cout << "ij2 " << ij << endl;
    ps.NewPage();

    c1->cd(1);
    h_correlht[ij]->Draw("colz");
    //    c1->cd(2); h_correlhten[ij][ij]->Draw("colz");
    c1->cd(2);
    h_rmoccu[ij]->Draw("colz");
    c1->cd(3);
    h_rmcorrel[ij]->Draw("colz");
    cout << "ij23 " << ij << endl;
    for (int jk = 0; jk < nringmx; jk++) {
      cout << "ij22 " << ij << endl;
      c1->cd(jk + 4);
      h_correlsig[jk][ij]->Draw("colz");
    }
    cout << "ij21 " << ij << endl;
    c1->Update();

    sprintf(name, "hoCalibc/hoallcorrelsig_%i", ij);
    cout << "ij24 " << ij << " " << name << endl;
    h_allcorrelsig[ij] = (TH2F*)fx->Get(name);
    h_allcorrelsig[ij]->Scale(1. / nentry);
    cout << "ij25 " << ij << endl;
    ps.NewPage();
    c3->cd();
    h_allcorrelsig[ij]->Draw("colz");
    c3->Update();
    cout << "ij2 " << ij << endl;
  }

  for (int jk = 0; jk < nringmx; jk++) {
    ps.NewPage();
    for (int kl = 1; kl <= routmx; kl++) {
      if (jk != 2 && kl > rout12mx)
        continue;
      c5->cd(kl);
      gPad->SetLogy(1);
      if (kl == 0) {
        sprintf(name, "hoCalibc/rout_ind_energy_r%i_allrout", jk);
      } else {
        sprintf(name, "hoCalibc/rout_ind_energy_r%i_%i", jk, kl);
      }
      rout_ind_energy[jk][kl] = (TH1F*)fx->Get(name);
      rout_ind_energy[jk][kl]->Draw();
    }
    c5->Update();
  }

  for (int ij = thmn; ij < thmx; ij++) {
    //    int klx=0;
    for (int jk = 0; jk < nringmx; jk++) {
      ps.NewPage();
      for (int kl = 1; kl <= routmx; kl++) {
        if (jk != 2 && kl > rout12mx)
          continue;
        c5->cd(kl);
        gPad->SetLogy(1);
        if (kl == 0) {
          sprintf(name, "hoCalibc/rout_mult_%i_r%i_allrout", ij, jk);
        } else {
          sprintf(name, "hoCalibc/rout_mult_%i_r%i_%i", ij, jk, kl);
        }
        rout_mult[ij][jk][kl] = (TH1F*)fx->Get(name);
        rout_mult[ij][jk][kl]->Draw();
      }
      c5->Update();
    }
  }

  for (int ij = thmn; ij < thmx; ij++) {
    int klx = 0;
    for (int jk = 0; jk < nringmx; jk++) {
      if (klx % 12 == 0)
        ps.NewPage();
      for (int kl = 1; kl <= rbxmx; kl++) {
        if (jk != 2 && kl > rbx12mx)
          continue;
        c4x->cd(klx % 12 + 1);
        klx++;
        gPad->SetLogy(1);
        if (kl == 0) {
          sprintf(name, "hoCalibc/rbx_mult_%i_r%i_allrbx", ij, jk);
        } else {
          sprintf(name, "hoCalibc/rbx_mult_%i_r%i_%i", ij, jk, kl);
        }
        rbx_mult[ij][jk][kl] = (TH1F*)fx->Get(name);
        rbx_mult[ij][jk][kl]->Draw();
      }
      if (klx % 12 == 0)
        c4x->Update();
    }
  }

  for (int ij = thmn; ij < thmx; ij++) {
    for (int jk = 0; jk < nprojtype + 1; jk++) {
      sprintf(name, "hoCalibc/endigisig_%i_%i", jk, ij);
      cout << "name " << name << endl;
      endigisig[jk][ij] = (TH1F*)fx->Get(name);
      sprintf(name, "hoCalibc/endigicnt_%i_%i", jk, ij);
      endigicnt[jk][ij] = (TH1F*)fx->Get(name);

      sprintf(name, "hoCalibc/rmdigisig_%i_%i", jk, ij);
      rmdigisig[jk][ij] = (TH2F*)fx->Get(name);
      sprintf(name, "hoCalibc/rmdigicnt_%i_%i", jk, ij);
      rmdigicnt[jk][ij] = (TH2F*)fx->Get(name);

      sprintf(name, "hoCalibc/ringdigisig_%i_%i", jk, ij);
      ringdigisig[jk][ij] = (TH2F*)fx->Get(name);
      sprintf(name, "hoCalibc/ringdigicnt_%i_%i", jk, ij);
      ringdigicnt[jk][ij] = (TH2F*)fx->Get(name);

      sprintf(name, "hoCalibc/inddigisig_%i_%i", jk, ij);
      inddigisig[jk][ij] = (TH2F*)fx->Get(name);
      sprintf(name, "hoCalibc/inddigicnt_%i_%i", jk, ij);
      inddigicnt[jk][ij] = (TH2F*)fx->Get(name);

      sprintf(name, "hoCalibc/indrecosig_%i_%i", jk, ij);
      indrecosig[jk][ij] = (TH2F*)fx->Get(name);
      sprintf(name, "hoCalibc/indrecocnt_%i_%i", jk, ij);
      indrecocnt[jk][ij] = (TH2F*)fx->Get(name);

      endigisig[jk][ij]->Divide(endigicnt[jk][ij]);
      rmdigisig[jk][ij]->Divide(rmdigicnt[jk][ij]);
      ringdigisig[jk][ij]->Divide(ringdigicnt[jk][ij]);
      inddigisig[jk][ij]->Divide(inddigicnt[jk][ij]);
      indrecosig[jk][ij]->Divide(indrecocnt[jk][ij]);
    }
  }

  ps.NewPage();

  for (int ij = thmn; ij < thmx; ij++) {
    ps.NewPage();
    for (int jk = 0; jk < nprojtype + 1; jk++) {
      c4->cd(2 * jk + 1);
      endigicnt[jk][ij]->Draw();
      c4->cd(2 * jk + 2);
      endigisig[jk][ij]->Draw();
    }
    c4->Update();
  }

  for (int ij = thmn; ij < thmx; ij++) {
    ps.NewPage();
    for (int jk = 0; jk < nprojtype + 1; jk++) {
      c4->cd(2 * jk + 1);
      rmdigicnt[jk][ij]->Draw("colz");
      c4->cd(2 * jk + 2);
      rmdigisig[jk][ij]->Draw("colz");
    }
    c4->Update();
  }

  for (int ij = thmn; ij < thmx; ij++) {
    ps.NewPage();
    for (int jk = 0; jk < nprojtype + 1; jk++) {
      c4->cd(2 * jk + 1);
      ringdigicnt[jk][ij]->Draw("colz");
      c4->cd(2 * jk + 2);
      ringdigisig[jk][ij]->Draw("colz");
    }
    c4->Update();
  }

  for (int ij = thmn; ij < thmx; ij++) {
    ps.NewPage();
    for (int jk = 0; jk < nprojtype + 1; jk++) {
      c4->cd(2 * jk + 1);
      inddigicnt[jk][ij]->Draw("colz");
      c4->cd(2 * jk + 2);
      inddigisig[jk][ij]->Draw("colz");
    }
    c4->Update();
  }

  for (int ij = thmn; ij < thmx; ij++) {
    ps.NewPage();
    for (int jk = 0; jk < nprojtype + 1; jk++) {
      c4->cd(2 * jk + 1);
      indrecocnt[jk][ij]->Draw("colz");
      c4->cd(2 * jk + 2);
      indrecosig[jk][ij]->Draw("colz");
    }
    c4->Update();
  }

  for (int ij = thmn; ij < thmx; ij++) {
    sprintf(name, "hoCalibc/hoallmucorrel_%i", ij);

    double total = 0;
    h_allmucorrel[ij] = (TH2F*)fx->Get(name);
    hnorm_allmucorrel[ij] = (TH2F*)h_allmucorrel[ij]->Clone();
    for (int ix = 1; ix <= h_allmucorrel[ij]->GetNbinsX(); ix++) {
      float anent = h_allmucorrel[ij]->GetBinContent(ix, 0);
      total += anent;
      cout << "ix " << ij << " " << ix << " " << anent << " " << total << endl;
      if (anent < 1.)
        anent = 1.;
      for (int iy = 1; iy <= h_allmucorrel[ij]->GetNbinsY(); iy++) {
        hnorm_allmucorrel[ij]->SetBinContent(ix, iy, h_allmucorrel[ij]->GetBinContent(ix, iy) / anent);
      }
    }
    ps.NewPage();
    c3->cd();
    hnorm_allmucorrel[ij]->Draw("colz");
    c3->Update();
  }

  cout << "ij116 " << endl;

  for (int isel = 0; isel < nseltype; isel++) {
    for (int ij = 0; ij < nprojmx / 2; ij++) {
      ps.NewPage();
      for (int jk = 0; jk < nadmx / 2; jk++) {
        sprintf(name, "hoCalibc/%s_%i_%i", projname[ij], isel, jk);
        cout << "name " << name << endl;
        totalproj[isel][ij][jk] = (TH2F*)fx->Get(name);
      }
    }
    cout << "1ij116 " << endl;
    for (int jk = 0; jk < nadmx / 2; jk++) {
      totalproj[isel][1][jk]->Divide(totalproj[isel][0][jk]);
      totalproj[isel][0][jk]->Divide(totalmuon[isel]);
    }

    for (int ij = 0; ij < nprojmx / 2; ij++) {
      ps.NewPage();
      for (int jk = 0; jk < nadmx / 2; jk++) {
        c1->cd(jk + 1);
        totalproj[isel][ij][jk]->Draw("colz");
      }
      c1->Update();
    }

    cout << "2ij116 " << endl;
    for (int ij = 2; ij < nprojmx; ij++) {
      for (int jk = 0; jk < nadmx; jk++) {
        sprintf(name, "hoCalibc/%s_%i_%i", projname[ij], isel, jk);
        cout << "ij2 " << ij << " " << jk << " " << name << endl;
        totalproj[isel][ij][jk] = (TH2F*)fx->Get(name);
      }
    }
    cout << "3ij116 " << endl;
    for (int jk = 0; jk < nadmx / 2; jk++) {
      totalproj[isel][3][jk]->Divide(totalproj[isel][2][jk]);
      totalproj[isel][2][jk]->Divide(totalmuon[isel]);
    }
    cout << "4ij116 " << endl;

    for (int ij = 2; ij < nprojmx; ij++) {
      ps.NewPage();
      for (int jk = 0; jk < nadmx; jk++) {
        c2->cd(jk + 1);
        totalproj[isel][ij][jk]->Draw("colz");
      }
      c2->Update();
    }

    cout << "5ij116 " << endl;
  }

  ps.Close();
}

void testxx(int nproj = 1) {
  def_setting();
  const int nprojtype = 5;  // Varities due to types of muon projection
  TH2F* histent[nprojtype + 3];
  TH2F* histen[nprojtype + 3];
  TH2F* histen2[nprojtype + 3];
  TH2F* histerr[nprojtype];
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.10);
  gStyle->SetPadRightMargin(0.12);
  gStyle->SetLabelSize(0.065, "XY");

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.07);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  cout << "testxxxx1 " << endl;
  TCanvas* c1 = new TCanvas("c1", "c1", 800, 550);
  cout << "testxxxx2 " << endl;
  c1->Divide(2, 2);
  //  TFile* fx = new TFile("histall_apr14b_cosmic_csa14_cosmic.root", "read");
  char name[100];

  for (int ij = 0; ij < nproj; ij++) {
    sprintf(name, "hoCalibc/hbentry_d%i", ij + 3);
    cout << "ij " << name << endl;
    histent[ij] = (TH2F*)gDirectory->Get(name);
    sprintf(name, "hoCalibc/hbsig_d%i", ij + 3);
    histen[ij] = (TH2F*)gDirectory->Get(name);
    sprintf(name, "hoCalibc/hbsig2_d%i", ij + 3);
    histen2[ij] = (TH2F*)gDirectory->Get(name);

    int nentry = TMath::Max(1., histent[ij]->GetBinContent(0, 0));
    histen[ij]->Divide(histent[ij]);
    histen2[ij]->Divide(histent[ij]);

    histent[ij]->Scale(1. / nentry);

    histen[ij]->GetXaxis()->SetLabelSize(0.065);
    histen[ij]->GetYaxis()->SetLabelSize(0.065);
    histen2[ij]->GetXaxis()->SetLabelSize(0.065);
    histen2[ij]->GetYaxis()->SetLabelSize(0.065);
    histent[ij]->GetXaxis()->SetLabelSize(0.065);
    histent[ij]->GetYaxis()->SetLabelSize(0.065);

    int xbin = histen[ij]->GetNbinsX();
    int ybin = histen[ij]->GetNbinsY();
    sprintf(name, "histerr_%i", ij);  //, indexx[ij]);
    histerr[ij] = new TH2F(name,
                           name,
                           xbin,
                           histent[ij]->GetXaxis()->GetXmin(),
                           histent[ij]->GetXaxis()->GetXmax(),
                           ybin,
                           histent[ij]->GetYaxis()->GetXmin(),
                           histent[ij]->GetYaxis()->GetXmax());

    for (int jk = 1; jk <= xbin; jk++) {
      for (int kl = 1; kl <= ybin; kl++) {
        double err = sqrt(histen2[ij]->GetBinContent(jk, kl) -
                          histen[ij]->GetBinContent(jk, kl) * histen[ij]->GetBinContent(jk, kl));

        histerr[ij]->Fill(histent[ij]->GetXaxis()->GetBinCenter(jk), histent[ij]->GetYaxis()->GetBinCenter(kl), err);
      }
    }

    cout << "ij11 " << ij << " " << histent[ij]->GetBinContent(22, 22) << endl;
    histent[ij]->GetXaxis()->SetTitle("i#eta");
    histent[ij]->GetXaxis()->SetTitleSize(0.075);
    histent[ij]->GetXaxis()->SetTitleOffset(0.8);
    histent[ij]->GetXaxis()->CenterTitle();
    histent[ij]->GetYaxis()->SetTitle("i#phi");
    histent[ij]->GetYaxis()->SetTitleSize(0.075);
    histent[ij]->GetYaxis()->SetTitleOffset(0.6);
    histent[ij]->GetYaxis()->CenterTitle();

    c1->cd(1);
    histent[ij]->Draw("colz");
    latex.DrawLatex(0.5, 0.95, "Occupancy");
    cout << "ij12 " << ij << " " << histen[ij]->GetBinContent(22, 22) << endl;
    histen[ij]->GetXaxis()->SetTitle("i#eta");
    histen[ij]->GetXaxis()->SetTitleSize(0.075);
    histen[ij]->GetXaxis()->SetTitleOffset(0.8);
    histen[ij]->GetXaxis()->CenterTitle();
    histen[ij]->GetYaxis()->SetTitle("i#phi");
    histen[ij]->GetYaxis()->SetTitleSize(0.075);
    histen[ij]->GetYaxis()->SetTitleOffset(0.6);
    histen[ij]->GetYaxis()->CenterTitle();

    c1->cd(2);
    histen[ij]->Draw("colz");
    latex.DrawLatex(0.5, 0.95, "Mean");
    histen2[ij]->GetXaxis()->SetTitle("i#eta");
    histen2[ij]->GetXaxis()->SetTitleSize(0.075);
    histen2[ij]->GetXaxis()->SetTitleOffset(0.8);
    histen2[ij]->GetXaxis()->CenterTitle();
    histen2[ij]->GetYaxis()->SetTitle("i#phi");
    histen2[ij]->GetYaxis()->SetTitleSize(0.075);
    histen2[ij]->GetYaxis()->SetTitleOffset(0.6);
    histen2[ij]->GetYaxis()->CenterTitle();

    c1->cd(3);
    histen2[ij]->Draw("colz");
    latex.DrawLatex(0.5, 0.95, "(Mean^{2})");
    cout << "ij14 " << ij << endl;
    histerr[ij]->GetXaxis()->SetTitle("i#eta");
    histerr[ij]->GetXaxis()->SetTitleSize(0.075);
    histerr[ij]->GetXaxis()->SetTitleOffset(0.8);
    histerr[ij]->GetXaxis()->CenterTitle();
    histerr[ij]->GetYaxis()->SetTitle("i#phi");
    histerr[ij]->GetYaxis()->SetTitleSize(0.075);
    histerr[ij]->GetYaxis()->SetTitleOffset(0.6);
    histerr[ij]->GetYaxis()->CenterTitle();

    c1->cd(4);
    histerr[ij]->Draw("colz");
    latex.DrawLatex(0.5, 0.95, "RMS");

    cout << "ij15 " << ij << endl;
    c1->Update();
    cout << "ij16 " << ij << endl;

    int nentry = TMath::Max(1., histent[ij]->GetBinContent(0, 0));
    cout << "ij " << nentry << endl;
  }
}

// plot000(-1,6,7,5)
// plot000(0,6,7,5)
// plot000(2,12,13,5)
// plot000(2,1,2,5)
// plot000(-1,1,2,5)
// plot000(3)
// plot000(2,5,10,5)

void plot000(int irng = -1, int isect = 0, int isect2 = 5, int ntype = 1, const char* tag = "", int extcol = 0) {
  def_setting();
  gStyle->SetOptLogy(1);
  gStyle->SetOptStat(0);
  gStyle->SetStatTextColor(3);
  gStyle->SetStatY(.99);
  gStyle->SetPadLeftMargin(0.12);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.12);
  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.07);
  //  latex.SetTextColor(extcol+1);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right
  /*
  char  pch1[200];
  cout<<" namec "<<gDirectory->GetName()<<endl;
  char* namex = gDirectory->GetName();
  int len =strstr(namex, ".");
  int len1 =strstr(namex, "\\"); 
  int len2=strlen(namex);

  strncpy (pch1, namex+17, len2-22);
  cout <<"name "<< namex<<" "<<len<<" "<<len1<<" "<<len2<<" "<<pch1<<endl;
  */

  TLegend* tleg = new TLegend(.5, 0.65, 0.75, 0.85, "", "brNDC");
  tleg->SetFillColor(10);
  tleg->SetBorderSize(0);
  tleg->SetTextFont(42);

  char name[100];
  if (!strstr(tag, "same")) {
    TCanvas* c1 = new TCanvas("c1", "c1", 1000, 600);
    if (irng >= 0 && isect2 - isect < 2) {
      c1->Divide(5, 2, 1.e-6, 1.e-6);
      tleg->SetTextSize(0.06);
    } else {
      if (irng < 0) {
        c1->Divide(nchnmx, ringmx, 1.e-6, 1.e-6);
      } else {
        c1->Divide(nchnmx, ringmx + 1, 1.e-6, 1.e-6);
      }
      tleg->SetTextSize(0.10);
    }
  }

  TH1F* histx[nprojtype][ringmx][nchnmx];
  TH1F* histy[nprojtype][ringmx][nchnmx];

  int nloop = (irng < 0) ? ringmx : isect2 - isect + 1;
  cout << "nloop " << nloop << endl;
  bool iscomb = false;  //For combined sect
  //  for (int kl=0; kl<nprojtype; kl++) {
  for (int kl = 0; kl < ntype; kl++) {
    int ipad = 0;
    int icol = extcol + kl + 1;
    if (icol >= 5)
      icol++;
    if (icol >= 10)
      icol++;
    if (extcol > 0)
      latex.SetTextColor(icol);
    for (int ij = 0; ij < nloop; ij++) {
      for (int jk = 0; jk < nchnmx; jk++) {
        if (irng < 0) {
          sprintf(name, "hoCalibc/ringavedigi_%i_%i_%i_%i", kl, ij, (ij == 2) ? rbxmx : rbx12mx, jk);
          iscomb = true;
        } else {
          sprintf(name, "hoCalibc/ringavedigi_%i_%i_%i_%i", kl, irng, isect + ij, jk);
          if ((irng == 2 && isect + ij == rbxmx) || (irng != 2 && isect + ij == rbx12mx))
            iscomb = true;
        }
        //      cout<<"ij "<<ij<<" "<<jk<<" "<<name<<endl;
        histy[kl][ij][jk] = (TH1F*)gDirectory->Get(name);
        histx[kl][ij][jk] = (TH1F*)histy[kl][ij][jk]->Clone();
        histx[kl][ij][jk]->Rebin(3);
        histx[kl][ij][jk]->Scale(1. / TMath::Max(1., histx[kl][ij][jk]->Integral()));
        c1->cd(++ipad);
        histx[kl][ij][jk]->GetXaxis()->SetTitle("HO digi");
        histx[kl][ij][jk]->GetXaxis()->SetTitleSize(0.075);
        histx[kl][ij][jk]->GetXaxis()->SetTitleOffset(0.7);  //0.85
        histx[kl][ij][jk]->GetXaxis()->CenterTitle();
        histx[kl][ij][jk]->GetXaxis()->SetLabelSize(0.075);
        histx[kl][ij][jk]->GetXaxis()->SetLabelOffset(-0.005);

        histx[kl][ij][jk]->GetYaxis()->SetTitle();
        histx[kl][ij][jk]->GetYaxis()->SetTitleSize(0.085);
        histx[kl][ij][jk]->GetYaxis()->SetTitleOffset(0.9);
        histx[kl][ij][jk]->GetYaxis()->CenterTitle();
        histx[kl][ij][jk]->GetYaxis()->SetLabelSize(0.065);
        histx[kl][ij][jk]->GetYaxis()->SetLabelOffset(0.01);

        histx[kl][ij][jk]->GetXaxis()->SetNdivisions(404);
        histx[kl][ij][jk]->GetYaxis()->SetNdivisions(404);

        histx[kl][ij][jk]->SetLineColor(icol);
        if (kl == 0) {
          //	  histx[kl][ij][jk]->GetXaxis()->SetRangeUser(-1,300);
          histx[kl][ij][jk]->Draw(tag);
          if (jk == 0) {
            if (irng < 0) {
              latex.DrawLatex(0.65, 0.62, Form("Ring %i", ij - 2));
            } else {
              if (iscomb) {
                latex.DrawLatex(0.65, 0.85 - 0.08 * extcol, Form("R %i", irng - 2));
              } else {
                latex.DrawLatex(0.65, 0.90 - 0.045 * extcol, Form("R%i Rbx%i", irng - 2, isect + ij));
              }
            }
          }
          if (ij == 0 && (!strstr(tag, "same")))
            latex.DrawLatex(0.45, 0.88, Form("TS %i", jk));
        } else {
          histx[kl][ij][jk]->Draw("same");
        }

        if (ij == 0 && jk == 0 && (!strstr(tag, "same"))) {
          tleg->AddEntry(histx[kl][ij][jk], projname[kl], "lpfe");
        }

        cout << "kl " << kl << "" << ij << "" << jk << " " << histx[kl][ij][jk]->Integral() << " "
             << histx[kl][ij][jk]->GetTitle() << endl;
      }
    }
  }
  if (!strstr(tag, "same")) {
    tleg->Draw();
  }
  if (strstr(tag, "sames")) {
    c1->Update();
  }
}

void plot000r(int irng = -1, int isup = 0, double xup = 200) {
  def_setting();
  gStyle->SetOptLogy(1);
  gStyle->SetOptStat(0);
  gStyle->SetStatTextColor(3);
  gStyle->SetStatY(.99);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.12);
  TLatex latex;
  latex.SetNDC();
  if (irng < 0) {
    latex.SetTextSize(0.08);
  } else {
    latex.SetTextSize(0.075);
  }
  //  latex.SetTextColor(extcol+1);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  TLegend* tleg = new TLegend(.5, 0.3, 0.8, 0.9, "", "brNDC");
  tleg->SetFillColor(10);
  tleg->SetBorderSize(0);
  tleg->SetTextFont(42);

  cout << "xx1x " << endl;
  char name[100];
  int ysiz = (irng < 0) ? 350 : 450;
  TCanvas* c1 = new TCanvas("c1", "c1", 900., ysiz);
  tleg->SetTextSize(0.035);
  if (irng < 0) {
    tleg->SetTextSize(0.045);
    c1->Divide(5, 1, 1.e-6, 1.e-6);
  } else if (irng == 2) {
    c1->Divide(4, 3, 1.e-6, 1.e-6);
  } else {
    c1->Divide(3, 2, 1.e-6, 1.e-6);
  }

  cout << "xxx " << endl;
  TH1F* histx[ringmx][routmx + 1];
  TH1F* histy[ringmx][routmx + 1];

  int nloop11 = (irng < 0) ? 0 : irng;
  int nloop12 = (irng < 0) ? ringmx : irng + 1;
  int nloop2 = (irng < 0 && isup == 0) ? 1 : ((irng == 2 || isup > 0) ? 36 : 24);  //12 : 6);

  bool iscomb = false;  //For combined sect
  cout << "xxx2 " << nloop11 << " " << nloop12 << " " << nloop2 << endl;
  int ipad = 0;
  for (int ij = nloop11; ij < nloop12; ij++) {
    if (isup > 0)
      c1->cd(++ipad);
    for (int jk = 0; jk < nloop2; jk++) {
      if (ij != 2 && jk >= 24)
        continue;
      cout << "xxx3 " << ij << " " << jk << endl;
      if (irng < 0 && isup == 0) {
        sprintf(name, "hoCalibc/rout_ind_energy_r%i_0", ij);
        iscomb = true;
      } else {
        sprintf(name, "hoCalibc/rout_ind_energy_r%i_%i", ij, jk + 1);
      }

      cout << "name " << name << endl;
      histy[ij][jk] = (TH1F*)gDirectory->Get(name);
      histx[ij][jk] = (TH1F*)histy[ij][jk]->Clone();
      //      histx[ij][jk]->Rebin(2);
      histx[ij][jk]->Scale(1. / TMath::Max(1., histx[ij][jk]->Integral()));
      if (isup <= 0)
        c1->cd(++ipad);
      histx[ij][jk]->GetXaxis()->SetTitle("HO RECO (GeV)");
      histx[ij][jk]->GetXaxis()->SetTitleSize(0.065);
      histx[ij][jk]->GetXaxis()->SetTitleOffset(0.85);  //0.85
      histx[ij][jk]->GetXaxis()->CenterTitle();
      histx[ij][jk]->GetXaxis()->SetLabelSize(0.07);
      histx[ij][jk]->GetXaxis()->SetLabelOffset(-0.003);

      histx[ij][jk]->GetYaxis()->SetTitle();
      histx[ij][jk]->GetYaxis()->SetTitleSize(0.085);
      histx[ij][jk]->GetYaxis()->SetTitleOffset(0.9);
      histx[ij][jk]->GetYaxis()->CenterTitle();
      histx[ij][jk]->GetYaxis()->SetLabelSize(0.07);
      histx[ij][jk]->GetYaxis()->SetLabelOffset(0.01);

      histx[ij][jk]->GetXaxis()->SetNdivisions(404);
      histx[ij][jk]->GetYaxis()->SetNdivisions(404);
      histx[ij][jk]->GetXaxis()->SetRangeUser(0., xup);
      //      histx[ij][jk]->SetMinimum(1.e-8);
      if (isup > 0)
        histx[ij][jk]->SetLineColor(jk + 1);
      histx[ij][jk]->Draw((jk > 0 && isup > 0) ? "same" : "");

      if (isup > 0 && ij == 2) {
        cout << "jk " << jk << endl;
        tleg->AddEntry(histx[ij][jk], Form("RM %i", jk), "lpfe");
      }

      if (irng < 0) {
        latex.DrawLatex(0.55, 0.92, Form("Ring %i", ij - 2));
      } else {
        latex.DrawLatex(0.55, 0.92, Form("Ring %i Sec%i", ij - 2, jk));
      }
    }
    if (isup > 0 && ij == 2)
      tleg->Draw();
  }
  //  if (!strstr(tag,"same")) tleg->Draw();
  c1->Update();
}

void plot00(int ityp = 0, int isect = 0) {
  def_setting();
  gStyle->SetOptLogy(1);
  gStyle->SetOptStat(1100);
  gStyle->SetStatTextColor(3);
  gStyle->SetStatY(.99);
  gStyle->SetPadLeftMargin(0.09);
  gStyle->SetPadRightMargin(0.02);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.07);
  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.11);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 1000., 600.);
  c1->Divide(nchnmx, ringmx, 1.e-6, 1.e-6);

  TH1F* histx[ringmx][nchnmx];
  int icol = 0;
  for (int ij = 0; ij < ringmx; ij++) {
    for (int jk = 0; jk < nchnmx; jk++) {
      //      sprintf(name, "hoCalibc/ringavedigi_%i_%i", ij, jk);
      sprintf(name, "hoCalibc/ringavedigi_%i_%i_%i_%i", ityp, ij, isect, jk);
      //      cout<<"ij "<<ij<<" "<<jk<<" "<<name<<endl;
      histx[ij][jk] = (TH1F*)gDirectory->Get(name);

      c1->cd(++icol);
      histx[ij][jk]->GetXaxis()->SetTitle("HO digi");
      histx[ij][jk]->GetXaxis()->SetTitleSize(0.095);
      histx[ij][jk]->GetXaxis()->SetTitleOffset(0.75);  //0.85
      histx[ij][jk]->GetXaxis()->CenterTitle();
      histx[ij][jk]->GetXaxis()->SetLabelSize(0.085);
      histx[ij][jk]->GetXaxis()->SetLabelOffset(0.001);

      histx[ij][jk]->GetYaxis()->SetTitle();
      histx[ij][jk]->GetYaxis()->SetTitleSize(0.085);
      histx[ij][jk]->GetYaxis()->SetTitleOffset(0.9);
      histx[ij][jk]->GetYaxis()->CenterTitle();
      histx[ij][jk]->GetYaxis()->SetLabelSize(0.07);
      histx[ij][jk]->GetYaxis()->SetLabelOffset(0.01);

      histx[ij][jk]->SetLineColor(3);
      histx[ij][jk]->Draw();
      if (jk == 0)
        latex.DrawLatex(0.65, 0.62, Form("Ring %i", ij - 2));
      if (ij == 0)
        latex.DrawLatex(0.65, 0.42, Form("TS %i", jk));
    }
  }
  c1->Update();
}

//root hocalib_cosmic_csa14_cosmic_set50_59.root
// root apr14/apr14b/hocalib_apr14b_cosmic_csa14_cosmic.root
// plot01(0) : //signal refion
// plot01(1) : //noise region
// plot01(2) : //total region
void plot01(int ityp = 0, int islocal = 1) {
  def_setting();
  gStyle->SetOptLogy(1);
  gStyle->SetOptStat(0);
  gStyle->SetStatTextColor(3);
  gStyle->SetStatY(.99);
  gStyle->SetPadRightMargin(0.02);

  gStyle->SetPadTopMargin(0.02);
  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.08);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 400.);
  c1->Divide(5, 1);

  TH1F* histx[ringmx][nprojtype];
  TH1F* histy[ringmx][nprojtype];

  int nloop = (islocal == 1) ? 1 : nprojtype;
  for (int jk = 0; jk < nloop; jk++) {
    for (int ij = 0; ij < ringmx; ij++) {
      sprintf(name, "hoCalibc/hoindringen_%i_%i", ij, jk);

      cout << "name " << name << endl;
      histy[ij][jk] = (TH1F*)gDirectory->Get(name);
      histx[ij][jk] = (TH1F*)histy[ij][jk]->Clone();
      histx[ij][jk]->Rebin(4);
      histx[ij][jk]->Scale(1. / histx[ij][jk]->Integral());
    }
  }
  for (int jk = 0; jk < nloop; jk++) {
    for (int ij = 0; ij < ringmx; ij++) {
      c1->cd(ij + 1);
      histx[ij][jk]->GetXaxis()->SetTitle((islocal) ? "Noise (GeV)" : "HO signal/noise (GeV)");
      histx[ij][jk]->GetXaxis()->SetTitleSize(0.075);
      histx[ij][jk]->GetXaxis()->SetTitleOffset(0.65);  //0.85
      histx[ij][jk]->GetXaxis()->CenterTitle();
      histx[ij][jk]->GetXaxis()->SetLabelSize(0.065);
      histx[ij][jk]->GetXaxis()->SetLabelOffset(-0.001);

      histx[ij][jk]->GetYaxis()->SetTitle();
      histx[ij][jk]->GetYaxis()->SetTitleSize(0.075);
      histx[ij][jk]->GetYaxis()->SetTitleOffset(0.9);
      histx[ij][jk]->GetYaxis()->CenterTitle();
      histx[ij][jk]->GetYaxis()->SetLabelSize(0.05);
      histx[ij][jk]->GetYaxis()->SetLabelOffset(0.01);

      histx[ij][jk]->SetLineColor(jk + 1);
      if (jk == 0) {
        double xmn = histx[ij][jk]->GetXaxis()->GetXmin();
        double xmx = histx[ij][jk]->GetXaxis()->GetXmax();
        if (ityp == 0) {
          histx[ij][jk]->GetXaxis()->SetRangeUser(xmn, 5.);
        } else if (ityp == 1) {
          histx[ij][jk]->GetXaxis()->SetRangeUser(2, xmx);
        } else {
          histx[ij][jk]->GetXaxis()->SetRangeUser(xmn, xmx);
        }
        //	histx[ij][jk]->Scale(1./histx[ij][jk]->GetEntries());
        histx[ij][jk]->Draw();
        latex.DrawLatex(0.25, 0.82, Form("Ring %i", ij - 2));
      } else {
        histx[ij][jk]->Draw("same");
      }
      //      cout<<"ijjk "<< ij<<" "<<jk<<" "<<histx[ij][jk]->GetEntries()<<" " <<histx[ij][jk]->Integral()<<" "<<histx[ij][jk]->GetTitle()<<endl;
    }
  }
  c1->Update();
}

//root hocalib_cosmic_csa14_cosmic_set50_59.root
// root apr14/apr14b/hocalib_apr14b_cosmic_csa14_cosmic.root
void plot1() {
  def_setting();
  gStyle->SetOptStat(1100);
  gStyle->SetStatTextColor(3);
  gStyle->SetStatY(.99);
  gStyle->SetPadRightMargin(0.02);

  gStyle->SetPadTopMargin(0.05);
  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.08);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(2, 2);

  TH1F* histx[8];
  for (int ij = 0; ij < 8; ij++) {
    sprintf(name, "signal_%i", ij);
    histx[ij] = new TH1F(name, name, 100, -0.2, 5.8);
    //  histx[1] = new TH1F("noise", "noise", 100, -.2, 5.8);

    //  TTree Ttree = gDirectory->Get("T1");
    switch (ij) {
      case 0:
        T1->Project(name, "hosig[4]", "isect>0&&(int(isect/100)-50)<-10&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
      case 1:
        T1->Project(name,
                    "hosig[4]",
                    "isect>0&&(int(isect/100)-50)>=-10&&(int(isect/100)-50)<-5&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
      case 2:
        T1->Project(name,
                    "hosig[4]",
                    "isect>0&&(int(isect/100)-50)>=-4&&(int(isect/100)-50)<5&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
      case 3:
        T1->Project(name,
                    "hosig[4]",
                    "isect>0&&(int(isect/100)-50)>=4&&(int(isect/100)-50)<11&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
      case 4:
        T1->Project(name, "hocro", "isect>0&&(int(isect/100)-50)<-10&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
      case 5:
        T1->Project(name,
                    "hocro",
                    "isect>0&&(int(isect/100)-50)>=-10&&(int(isect/100)-50)<-5&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
      case 6:
        T1->Project(name,
                    "hocro",
                    "isect>0&&(int(isect/100)-50)>=-4&&(int(isect/100)-50)<5&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
      case 7:
        T1->Project(name,
                    "hocro",
                    "isect>0&&(int(isect/100)-50)>=4&&(int(isect/100)-50)<11&&abs(hodx)>1&&abs(hody)>1&&ndof>20");
        break;
    }
  }
  for (int ij = 0; ij < 8; ij++) {
    c1->cd(ij % 4 + 1);
    histx[ij]->GetXaxis()->SetTitle("HO signal/noise (GeV)");
    histx[ij]->GetXaxis()->SetTitleSize(0.075);
    histx[ij]->GetXaxis()->SetTitleOffset(0.75);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.065);
    histx[ij]->GetXaxis()->SetLabelOffset(0.001);

    histx[ij]->GetYaxis()->SetTitle();
    histx[ij]->GetYaxis()->SetTitleSize(0.075);
    histx[ij]->GetYaxis()->SetTitleOffset(0.9);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.05);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    if (ij < 4) {
      histx[ij]->SetLineColor(3);
      histx[ij]->SetMaximum(1.1 * histx[ij + 4]->GetMaximum());
      histx[ij]->Draw();
      latex.DrawLatex(0.25, 0.82, Form("Ring %i", ij - 2));
    } else {
      if (ij == 4) {
        c1->Update();
        gStyle->SetStatTextColor(2);
        gStyle->SetStatY(.79);
      }
      histx[ij]->SetLineColor(2);
      histx[ij]->Draw("sames");
    }
    //    histx[1]->Draw("sames");
  }
  c1->Update();
}

//root hocalib_cosmic_csa14_cosmic_set50_59.root
// root apr14/apr14b/hocalib_apr14b_cosmic_csa14_cosmic.root
void plot1a() {
  def_setting();
  gStyle->SetOptStat(1100);
  gStyle->SetOptLogy(1);
  gStyle->SetStatTextColor(3);
  gStyle->SetStatY(.99);
  gStyle->SetPadRightMargin(0.02);
  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.08);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(2, 2);

  TH1F* histx[8];
  for (int ij = 0; ij < 8; ij++) {
    sprintf(name, "signal_%i", ij);
    histx[ij] = new TH1F(name, name, 100, 0.5, 19.5);
    //  histx[1] = new TH1F("noise", "noise", 100, -.2, 5.8);

    //  TTree Ttree = gDirectory->Get("Tcor1");
    switch (ij) {
      case 0:
        Tcor1->Project(name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)<-10&&int(icapid/10.)==4");
        break;
      case 1:
        Tcor1->Project(
            name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)>=-10&&(int(hocomu/100)-50)<-5&&int(icapid/10.)==4");
        break;
      case 2:
        Tcor1->Project(
            name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)>=-4&&(int(hocomu/100)-50)<5&&int(icapid/10.)==4");
        break;
      case 3:
        Tcor1->Project(
            name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)>=4&&(int(hocomu/100)-50)<11&&int(icapid/10.)==4");
        break;
      case 4:
        Tcor1->Project(name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)<-10&&int(icapid/10.)==0");
        break;
      case 5:
        Tcor1->Project(
            name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)>=-10&&(int(hocomu/100)-50)<-5&&int(icapid/10.)==0");
        break;
      case 6:
        Tcor1->Project(
            name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)>=-4&&(int(hocomu/100)-50)<5&&int(icapid/10.)==0");
        break;
      case 7:
        Tcor1->Project(
            name, "hocellen", "hocomu>0&&(int(hocomu/100)-50)>=4&&(int(hocomu/100)-50)<11&&int(icapid/10.)==0");
        break;
    }

    c1->cd(ij % 4 + 1);
    histx[ij]->GetXaxis()->SetTitle("HO signal/noise (GeV)");
    histx[ij]->GetXaxis()->SetTitleSize(0.075);
    histx[ij]->GetXaxis()->SetTitleOffset(0.75);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.065);
    histx[ij]->GetXaxis()->SetLabelOffset(0.001);

    histx[ij]->GetYaxis()->SetTitle();
    histx[ij]->GetYaxis()->SetTitleSize(0.075);
    histx[ij]->GetYaxis()->SetTitleOffset(0.9);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.055);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    if (ij < 4) {
      histx[ij]->SetLineColor(3);

      histx[ij]->Draw();
      latex.DrawLatex(0.25, 0.82, Form("Ring %i", ij - 2));

    } else {
      if (ij == 4) {
        c1->Update();
        gStyle->SetStatTextColor(2);
        gStyle->SetStatY(.79);
      }

      histx[ij]->SetLineColor(2);
      histx[ij]->Draw("sames");
    }
    //    histx[1]->Draw("sames");
  }
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
//plot2(1,4,0)

void plot2(int ith1 = 1, int ith2 = 4, int iproj = 0) {
  def_setting();
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.07);
  gStyle->SetPadBottomMargin(0.12);
  gStyle->SetPadLeftMargin(0.11);
  gStyle->SetPadRightMargin(0.14);

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.06);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  char name[100];
  TH2F* histx[10];
  TH2F* histy[10];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(2, 2);
  TH2F* histz = (TH2F*)gDirectory->Get("hoCalibc/hbentry_d3");
  int nentry = histz->GetBinContent(0, 0);

  icnt = 0;
  cout << "nentry = " << nentry << endl;
  for (int ij = ith1; ij <= ith2; ij++) {
    //  sprintf(name, "hoCalibc/indrecosig_%i_%i", iproj, ij);
    //  indrecosig[jk][ij] = (TH2F*)fx->Get(name);
    sprintf(name, "hoCalibc/indrecocnt_%i_%i", iproj, ij);
    histy[icnt] = (TH2F*)gDirectory->Get(name);
    histx[icnt] = (TH2F*)histy[icnt]->Clone();

    c1->cd(icnt + 1);
    cout << "icnt " << icnt << " " << histx[icnt]->GetTitle() << endl;
    histx[icnt]->Scale(1. / nentry);
    histx[icnt]->GetXaxis()->SetTitle("i#eta");
    histx[icnt]->GetXaxis()->SetTitleSize(0.075);
    histx[icnt]->GetXaxis()->SetTitleOffset(0.75);  //0.85
    histx[icnt]->GetXaxis()->CenterTitle();
    histx[icnt]->GetXaxis()->SetLabelSize(0.065);
    histx[icnt]->GetXaxis()->SetLabelOffset(0.001);

    histx[icnt]->GetYaxis()->SetTitle("i#phi");
    histx[icnt]->GetYaxis()->SetTitleSize(0.075);
    histx[icnt]->GetYaxis()->SetTitleOffset(0.7);
    histx[icnt]->GetYaxis()->CenterTitle();
    histx[icnt]->GetYaxis()->SetLabelSize(0.065);
    histx[icnt]->GetYaxis()->SetLabelOffset(0.01);

    histx[icnt]->GetZaxis()->SetLabelSize(0.048);
    histx[icnt]->GetZaxis()->SetLabelOffset(0.01);
    histx[icnt]->GetZaxis()->SetNdivisions(406);

    histx[icnt]->Draw("colz");
    latex.DrawLatex(.4, .95, Form("%g <E_{th} < %g", hothreshs[ij], hothreshs[ij + 1]));
    icnt++;
  }
  c1->Update();
}
// root histall_apr14b_cosmic_csa14_cosmic.root
//xx  plot3(0)
//xx  plot3(1)
//xx  plot3(2)
//xx  plot3(3)

void plot3(int isel = 0) {
  def_setting();
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadBottomMargin(0.1);
  gStyle->SetPadLeftMargin(0.08);
  gStyle->SetPadRightMargin(0.14);
  char name[100];
  TH2F* histx[9];
  TH2F* histy[9];

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.08);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  sprintf(name, "hoCalibc/totalmuon_%i", isel);
  TH2F* totalmuon = (TH2F*)gDirectory->Get(name);

  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(3, 3, 1.e-6, 1.e-6);
  for (int ij = 0; ij < 9; ij++) {
    sprintf(name, "hoCalibc/totalproj_%i_%i", isel, ij);
    histy[ij] = (TH2F*)gDirectory->Get(name);
    histx[ij] = (TH2F*)histy[ij]->Clone();

    histx[ij]->Divide(totalmuon);

    c1->cd(ij + 1);
    //   histx[ij]->GetXaxis()->SetTitle("i#eta");
    histx[ij]->GetXaxis()->SetTitleSize(0.075);
    histx[ij]->GetXaxis()->SetTitleOffset(0.75);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.065);
    histx[ij]->GetXaxis()->SetLabelOffset(0.001);

    //    histx[ij]->GetYaxis()->SetTitle("i#phi");
    histx[ij]->GetYaxis()->SetTitleSize(0.075);
    histx[ij]->GetYaxis()->SetTitleOffset(0.7);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.055);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    histx[ij]->GetZaxis()->SetLabelSize(0.055);
    histx[ij]->GetZaxis()->SetLabelOffset(0.01);
    histx[ij]->GetZaxis()->SetNdivisions(406);
    histx[ij]->Draw("colz");
  }
  c1->Update();
}
// root histall_apr14b_cosmic_csa14_cosmic.root
// plot4(1,3,0,0)
void plot4(int ith1 = 0, int ith2 = 8, int isel1 = 0, int isel2 = 0) {
  def_setting();
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetPadRightMargin(0.02);
  char name[100];
  TH1F* histx[10][9];
  TH1F* histy[10][9];
  TH1F* histz[10][9];

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.055);
  latex.SetTextFont(42);
  latex.SetTextAlign(3);  //(31); // align right

  int nydiv = (isel2 == isel1) ? 1 : 2;
  //  TCanvas* c1 = new TCanvas("c1", "c1", 800., (nydiv==1) ? 350 : 600.);
  //  c1->Divide(3,nydiv,1.e-6,1.e-6);
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 600.);
  c1->Divide(3, 3, 1.e-6, 1.e-6);

  const int nbin = 5;
  TString labels[nbin] = {"Ring-2", "Ring-1", "Ring0", "Ring+1", "Ring+2"};

  for (int ij = ith1; ij <= ith2; ij++) {
    sprintf(name, "hoCalibc/endigisig_%i_%i", isel1, ij);
    histz[0][ij] = (TH1F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/endigisig_%i_%i", isel2, ij);
    histz[1][ij] = (TH1F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/endigicnt_%i_%i", isel1, ij);
    histy[0][ij] = (TH1F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/endigicnt_%i_%i", isel2, ij);
    histy[1][ij] = (TH1F*)gDirectory->Get(name);
  }

  int icol = 0;
  for (int jk = 0; jk < nydiv; jk++) {
    for (int ij = ith1; ij <= ith2; ij++) {
      histx[jk][ij] = (TH1F*)histz[jk][ij]->Clone();
      histx[jk][ij]->Divide(histy[jk][ij]);
      c1->cd(++icol);
      cout << "ij " << icol << " " << ij << " " << jk << " " << histy[jk][ij]->GetTitle() << endl;

      histx[jk][ij]->GetXaxis()->SetLabelSize(0.05);
      histx[jk][ij]->GetXaxis()->SetLabelOffset(0.001);

      histx[jk][ij]->GetYaxis()->SetLabelSize(0.045);
      histx[jk][ij]->GetYaxis()->SetLabelOffset(0.01);

      histx[jk][ij]->GetXaxis()->SetTitle("Time slice in Five rings");
      histx[jk][ij]->GetXaxis()->SetTitleSize(0.055);
      histx[jk][ij]->GetXaxis()->SetTitleOffset(1.2);  //0.85
      histx[jk][ij]->GetXaxis()->CenterTitle();
      histx[jk][ij]->GetXaxis()->SetLabelSize(0.075);
      histx[jk][ij]->GetXaxis()->SetLabelOffset(0.01);

      histx[jk][ij]->GetYaxis()->SetTitle("Signal (fC/GeV)");
      histx[jk][ij]->GetYaxis()->SetTitleSize(0.055);
      histx[jk][ij]->GetYaxis()->SetTitleOffset(1.3);
      histx[jk][ij]->GetYaxis()->CenterTitle();
      histx[jk][ij]->GetYaxis()->SetLabelSize(0.065);
      histx[jk][ij]->GetYaxis()->SetLabelOffset(0.01);

      histx[jk][ij]->GetXaxis()->SetNdivisions(406);
      histx[jk][ij]->GetYaxis()->SetNdivisions(406);

      for (int kl = 0; kl < nbin; kl++) {
        histx[jk][ij]->GetXaxis()->SetBinLabel(10 * kl + 6, labels[kl]);
      }
      histx[jk][ij]->GetXaxis()->LabelsOption("h");
      histx[jk][ij]->Draw();

      latex.DrawLatex(0.40, 0.93, Form("%g <E_{RECO} <%g GeV", hothreshs[ij], hothreshs[ij + 1]));
    }
  }
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
// plot5(1,3,0,0)
void plot5(int ith1 = 1, int ith2 = 3, int isel1 = 0, int isel2 = 4, double amx = -1.0) {
  def_setting();
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetPadRightMargin(0.14);

  char name[100];
  TH2F* histx[5][9];
  TH2F* histy[5][9];
  TH2F* histz[5][9];

  const int nbin = 5;
  TString labels[nbin] = {"Ring-2", "Ring-1", "Ring0", "Ring+1", "Ring+2"};

  int nydiv = (isel1 == isel2) ? 1 : 2;
  TCanvas* c1 = new TCanvas("c1", "c1", 800., (nydiv == 1) ? 350 : 600.);
  c1->Divide(3, nydiv, 1.e-6, 1.e-6);

  for (int ij = ith1; ij <= ith2; ij++) {
    sprintf(name, "hoCalibc/rmdigisig_%i_%i", isel1, ij);
    histz[0][ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/rmdigisig_%i_%i", isel2, ij);
    histz[1][ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/rmdigicnt_%i_%i", isel1, ij);
    histy[0][ij] = (TH2F*)gDirectory->Get(name);

    sprintf(name, "hoCalibc/rmdigicnt_%i_%i", isel2, ij);
    histy[1][ij] = (TH2F*)gDirectory->Get(name);
  }

  int icol = 0;
  for (int jk = 0; jk < nydiv; jk++) {
    for (int ij = ith1; ij <= ith2; ij++) {
      histx[jk][ij] = (TH2F*)histz[jk][ij]->Clone();
      histx[jk][ij]->Divide(histy[jk][ij]);
      c1->cd(++icol);
      cout << "ij " << icol << " " << ij << " " << jk << " " << histy[jk][ij]->GetTitle() << endl;

      histx[jk][ij]->GetXaxis()->SetTitle("Time slice in Five rings");
      histx[jk][ij]->GetXaxis()->SetTitleSize(0.055);
      histx[jk][ij]->GetXaxis()->SetTitleOffset(1.2);  //0.85
      histx[jk][ij]->GetXaxis()->CenterTitle();
      histx[jk][ij]->GetXaxis()->SetLabelSize(0.065);
      histx[jk][ij]->GetXaxis()->SetLabelOffset(0.01);

      histx[jk][ij]->GetYaxis()->SetTitle("RM#");
      histx[jk][ij]->GetYaxis()->SetTitleSize(0.055);
      histx[jk][ij]->GetYaxis()->SetTitleOffset(1.3);
      histx[jk][ij]->GetYaxis()->CenterTitle();
      histx[jk][ij]->GetYaxis()->SetLabelSize(0.065);
      histx[jk][ij]->GetYaxis()->SetLabelOffset(0.01);

      histx[jk][ij]->GetZaxis()->SetLabelSize(0.048);
      histx[jk][ij]->GetZaxis()->SetLabelOffset(0.01);

      histx[jk][ij]->GetXaxis()->SetNdivisions(202);
      //      histx[jk][ij]->GetYaxis()->SetNdivisions(406);
      //      histx[jk][ij]->GetZaxis()->SetNdivisions(406);
      if (amx > 0)
        histx[jk][ij]->SetMaximum(amx);
      //      for (int kl=0; kl<nbin; kl++) {
      // 	histx[jk][ij]->GetXaxis()->SetBinLabel(10*kl+6, labels[kl]);
      //       }
      //       histx[jk][ij]->GetXaxis()->LabelsOption("h");
      histx[jk][ij]->GetXaxis()->SetNdivisions(810);
      histx[jk][ij]->Draw("colz");
    }
  }
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
// plot6(-2,3)
void plot6(int iring = -2, int irbx = 3) {
  def_setting();
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetPadBottomMargin(0.08);
  gStyle->SetPadLeftMargin(0.20);
  gStyle->SetPadRightMargin(0.20);
  char name[100];
  const int shapemx = 10;
  TH2F* histx[shapemx];
  //  const char* shape_name[shapemx]={"N:T3/T34", "N:T34/Sum2-5", "N:T3/Sum2-5", "N:T4/Sum2-5", "N:Oth/Sum2-5",
  //				   "S:T3/T34", "S:T34/Sum2-5", "S:T3/Sum2-5", "S:T4/Sum2-5", "S:Oth/Sum2-5"};

  const char* shape_name[shapemx] = {
      "T3/T4", "T4/Sum45", "T45/Sum4-7", "T4-7/All", "T0-3/T4-7", "T0-3/All", "T0-3/T4-7", "T67/T45", "xx", "xx"};

  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  c1->Divide(4, 2, 1.e-6, 1.e-6);

  for (int ij = 0; ij < shapemx - 2; ij++) {
    c1->cd(ij + 1);
    sprintf(name, "hoCalibc/rbx_shape_%i_r%i_%i", ij, iring + 2, irbx);
    histx[ij] = (TH2F*)gDirectory->Get(name);
    cout << "ij " << ij << " " << histx[ij]->GetTitle() << endl;

    if (ij == 0 || ij == 5) {
      //      histx[ij]->GetXaxis()->SetTitle("T2+T3");
    } else {
      //      histx[ij]->GetXaxis()->SetTitle("Sum T2-5");
    }
    histx[ij]->GetXaxis()->SetTitleSize(0.065);
    histx[ij]->GetXaxis()->SetTitleOffset(0.65);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.065);
    histx[ij]->GetXaxis()->SetLabelOffset(-0.01);

    histx[ij]->GetYaxis()->SetTitle(shape_name[ij]);
    histx[ij]->GetYaxis()->SetTitleSize(0.065);
    histx[ij]->GetYaxis()->SetTitleOffset(1.3);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.065);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    histx[ij]->GetZaxis()->SetLabelSize(0.07);
    histx[ij]->GetZaxis()->SetLabelOffset(0.01);

    histx[ij]->GetXaxis()->SetNdivisions(406);
    histx[ij]->GetYaxis()->SetNdivisions(406);
    histx[ij]->GetZaxis()->SetNdivisions(406);

    histx[ij]->Draw("colz");
  }
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
// plot7(2,0)
void plot7(int ith = 2, int irbx = 0) {
  def_setting();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.08);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  const int nringmx = 5;
  gStyle->SetOptLogy(1);
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadRightMargin(0.02);
  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  c1->Divide(5, 2, 1.e-6, 1.e-6);

  TH1F* histx[2][nringmx];

  int icol = 0;

  for (int ij = 0; ij < 2; ij++) {
    for (int jk = 0; jk < nringmx; jk++) {
      if (ij == 0) {
        sprintf(name, "hoCalibc/rbx_mult_%i_r%i_%i", ith, jk, irbx);
      } else {
        sprintf(name, "hoCalibc/rout_mult_%i_r%i_%i", ith, jk, irbx);
      }

      histx[ij][jk] = (TH1F*)gDirectory->Get(name);
      cout << "name " << name << endl;
      c1->cd(++icol);

      if (ij == 0) {
        histx[ij][jk]->GetXaxis()->SetTitle("# in RBX");
      } else {
        histx[ij][jk]->GetXaxis()->SetTitle("# in RM");
      }
      histx[ij][jk]->GetXaxis()->SetTitleSize(0.085);
      histx[ij][jk]->GetXaxis()->SetTitleOffset(0.7);  //0.85
      histx[ij][jk]->GetXaxis()->CenterTitle();
      histx[ij][jk]->GetXaxis()->SetLabelSize(0.085);
      histx[ij][jk]->GetXaxis()->SetLabelOffset(-0.01);

      histx[ij][jk]->GetYaxis()->SetTitle();  //shape_name[ij][jk]);
      histx[ij][jk]->GetYaxis()->SetTitleSize(0.075);
      histx[ij][jk]->GetYaxis()->SetTitleOffset(1.3);
      histx[ij][jk]->GetYaxis()->CenterTitle();
      histx[ij][jk]->GetYaxis()->SetLabelSize(0.085);
      histx[ij][jk]->GetYaxis()->SetLabelOffset(0.01);

      histx[ij][jk]->GetXaxis()->SetNdivisions(404);
      histx[ij][jk]->GetYaxis()->SetNdivisions(404);

      histx[ij][jk]->Draw();
      cout << "ijjk " << ij << " " << jk << " " << histx[ij][jk]->GetTitle() << endl;
      latex.DrawLatex(0.35, 0.8, Form("Ring %i", jk - 2));
    }
  }
  c1->Update();
}

void plot7a(int irng = -1, int ith = 2, int isrm = 0) {
  def_setting();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.08);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  const int nringmx = 5;
  gStyle->SetOptLogy(1);
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadRightMargin(0.02);
  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  if (irng < 0) {
    c1->Divide(5, 2, 1.e-6, 1.e-6);
  } else if (isrm) {
    if (irng == 2) {
      c1->Divide(6, 6, 1.e-6, 1.e-6);
    } else {
      c1->Divide(6, 4, 1.e-6, 1.e-6);
    }
  } else {
    if (irng == 2) {
      c1->Divide(4, 3, 1.e-6, 1.e-6);
    } else {
      c1->Divide(3, 2, 1.e-6, 1.e-6);
    }
  }
  TH1F* histx[routmx + 1][routmx + 1];

  int icol = 0;

  int nloop1 = 2;
  int nloop2 = nringmx;
  if (irng >= 0) {
    nloop1 = 1;
    if (isrm) {
      nloop2 = (irng == 2) ? 36 : 24;
    } else {
      nloop2 = (irng == 2) ? 12 : 6;
    }
  }

  for (int ij = 0; ij < nloop1; ij++) {    //ij<2
    for (int jk = 0; jk < nloop2; jk++) {  //ij<nringmx;

      if (irng < 0) {
        if (ij == 0) {
          sprintf(name, "hoCalibc/rbx_mult_%i_r%i_0", ith, jk);  //, irbx);
        } else {
          sprintf(name, "hoCalibc/rout_mult_%i_r%i_0", ith, jk);  //, irbx);
        }
      } else if (isrm) {
        sprintf(name, "hoCalibc/rout_mult_%i_r%i_%i", ith, irng, jk);
      } else {
        sprintf(name, "hoCalibc/rbx_mult_%i_r%i_%i", ith, irng, jk);
      }

      histx[ij][jk] = (TH1F*)gDirectory->Get(name);
      cout << "name " << name << endl;
      c1->cd(++icol);
      if (irng < 0) {
        if (ij == 0) {
          histx[ij][jk]->GetXaxis()->SetTitle("# in RBX");
        } else {
          histx[ij][jk]->GetXaxis()->SetTitle("# in RM");
        }
      } else if (isrm) {
        histx[ij][jk]->GetXaxis()->SetTitle("# in RM");
      } else {
        histx[ij][jk]->GetXaxis()->SetTitle("# in RBX");
      }

      histx[ij][jk]->GetXaxis()->SetTitleSize(0.085);
      histx[ij][jk]->GetXaxis()->SetTitleOffset(0.7);  //0.85
      histx[ij][jk]->GetXaxis()->CenterTitle();
      histx[ij][jk]->GetXaxis()->SetLabelSize(0.085);
      histx[ij][jk]->GetXaxis()->SetLabelOffset(-0.01);

      histx[ij][jk]->GetYaxis()->SetTitle();  //shape_name[ij][jk]);
      histx[ij][jk]->GetYaxis()->SetTitleSize(0.075);
      histx[ij][jk]->GetYaxis()->SetTitleOffset(1.3);
      histx[ij][jk]->GetYaxis()->CenterTitle();
      histx[ij][jk]->GetYaxis()->SetLabelSize(0.085);
      histx[ij][jk]->GetYaxis()->SetLabelOffset(0.01);

      histx[ij][jk]->GetXaxis()->SetNdivisions(404);
      histx[ij][jk]->GetYaxis()->SetNdivisions(404);

      histx[ij][jk]->Draw();
      cout << "ijjk " << ij << " " << jk << " " << histx[ij][jk]->GetTitle() << endl;
      if (irng < 0) {
        latex.DrawLatex(0.35, 0.8, Form("Ring %i E>%g", jk - 2, hothreshs[ith]));
      } else if (isrm) {
        latex.DrawLatex(0.35, 0.8, Form("Ring %i RM%i E>%g", irng, jk, hothreshs[ith]));
      } else {
        latex.DrawLatex(0.35, 0.8, Form("Ring %i RBX%i E>%g", irng, jk, hothreshs[ith]));
      }
    }
  }
  c1->Update();
}

void plot7b(int irng = -1, int th1 = 2, int th2 = 5, int isrm = 0) {
  def_setting();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.065);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  TLegend* tleg = new TLegend(.75, 0.35, 0.95, 0.9, "", "brNDC");
  tleg->SetFillColor(10);
  tleg->SetBorderSize(0);
  tleg->SetTextFont(42);
  tleg->SetTextSize(0.07);

  const int nringmx = 5;
  gStyle->SetOptLogy(1);
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.02);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetPadRightMargin(0.02);
  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  if (irng < 0) {
    c1->Divide(5, 2, 1.e-6, 1.e-6);
  } else if (isrm) {
    if (irng == 2) {
      c1->Divide(6, 6, 1.e-6, 1.e-6);
    } else {
      c1->Divide(6, 4, 1.e-6, 1.e-6);
    }
  } else {
    if (irng == 2) {
      c1->Divide(4, 3, 1.e-6, 1.e-6);
    } else {
      c1->Divide(3, 2, 1.e-6, 1.e-6);
    }
  }
  TH1F* histx[routmx + 1][routmx + 1][nhothresh];

  int icol = 0;

  int nloop1 = 2;
  int nloop2 = nringmx;
  if (irng >= 0) {
    nloop1 = 1;
    if (isrm) {
      nloop2 = (irng == 2) ? 36 : 24;
    } else {
      nloop2 = (irng == 2) ? 12 : 6;
    }
  }

  for (int ij = 0; ij < nloop1; ij++) {    //ij<2
    for (int jk = 0; jk < nloop2; jk++) {  //ij<nringmx;
      for (int kl = th1; kl <= th2; kl++) {
        if (irng < 0) {
          if (ij == 0) {
            sprintf(name, "hoCalibc/rbx_mult_%i_r%i_0", kl, jk);  //, irbx);
          } else {
            sprintf(name, "hoCalibc/rout_mult_%i_r%i_0", kl, jk);  //, irbx);
          }
        } else if (isrm) {
          sprintf(name, "hoCalibc/rout_mult_%i_r%i_%i", kl, irng, jk + 1);
        } else {
          sprintf(name, "hoCalibc/rbx_mult_%i_r%i_%i", kl, irng, jk + 1);
        }

        histx[ij][jk][kl] = (TH1F*)gDirectory->Get(name);
        cout << "name " << name << endl;
        if (kl == th1)
          c1->cd(++icol);
        if (irng < 0) {
          if (ij == 0) {
            histx[ij][jk][kl]->GetXaxis()->SetTitle("# in RBX");
          } else {
            histx[ij][jk][kl]->GetXaxis()->SetTitle("# in RM");
          }
        } else if (isrm) {
          histx[ij][jk][kl]->GetXaxis()->SetTitle("# in RM");
        } else {
          histx[ij][jk][kl]->GetXaxis()->SetTitle("# in RBX");
        }

        histx[ij][jk][kl]->GetXaxis()->SetTitleSize(0.065);
        histx[ij][jk][kl]->GetXaxis()->SetTitleOffset(0.75);  //0.85
        histx[ij][jk][kl]->GetXaxis()->CenterTitle();
        histx[ij][jk][kl]->GetXaxis()->SetLabelSize(0.075);
        histx[ij][jk][kl]->GetXaxis()->SetLabelOffset(-0.01);

        histx[ij][jk][kl]->GetYaxis()->SetTitle();  //shape_name[ij][jk][kl]);
        histx[ij][jk][kl]->GetYaxis()->SetTitleSize(0.065);
        histx[ij][jk][kl]->GetYaxis()->SetTitleOffset(1.3);
        histx[ij][jk][kl]->GetYaxis()->CenterTitle();
        histx[ij][jk][kl]->GetYaxis()->SetLabelSize(0.075);
        histx[ij][jk][kl]->GetYaxis()->SetLabelOffset(0.01);

        histx[ij][jk][kl]->GetXaxis()->SetNdivisions(404);
        histx[ij][jk][kl]->GetYaxis()->SetNdivisions(404);

        histx[ij][jk][kl]->SetLineColor(kl - th1 + 1);
        histx[ij][jk][kl]->Draw((kl == th1) ? "" : "same");
        cout << "ijjk " << ij << " " << jk << " " << kl - th1 + 1 << " " << histx[ij][jk][kl]->GetTitle() << endl;
        if (kl == th1) {
          if (irng < 0) {
            latex.DrawLatex(0.35, 0.85, Form("R%i", jk - 2));
          } else if (isrm) {
            latex.DrawLatex(0.35, 0.85, Form("R%i RM%i", irng - 2, jk));
          } else {
            latex.DrawLatex(0.35, 0.85, Form("R%i RBX%i", irng - 2, jk));
          }
        }
        if (ij == 0 && jk == 0) {
          tleg->AddEntry(histx[ij][jk][kl], Form("th>%g", hothreshs[kl]), "lpfe");
        }
      }
    }
  }
  tleg->Draw();
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
// plot8(0,5,0)
// plot8(0,5,1)
void plot8(int ith1 = 1, int ith2 = 8, int ityp = 1) {
  def_setting();

  gStyle->SetPadGridX(0);
  gStyle->SetPadGridY(0);

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.06);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  const int nringmx = 5;
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadRightMargin(0.16);

  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  c1->Divide(4, 2, 1.e-6, 1.e-6);
  //  sprintf(name, "hoCalibc/hbentry_d%i", ij+3);
  int nentry = ((TH1F*)gDirectory->Get("hoCalibc/hbentry_d3"))->GetBinContent(0, 0);
  TH2F* histx[11];
  TH2F* histy[11];

  int icol = 0;
  for (int ij = ith1; ij <= ith2; ij++) {
    //    histx[1] = (TH2F*)gDirectory->Get(name);
    if (ityp == 0) {
      sprintf(name, "hoCalibc/hormoccu_%i", ij);
    } else {
      sprintf(name, "hoCalibc/hocorrelht_%i", ij);
    }
    cout << "nentry " << nentry << endl;
    histy[ij] = (TH2F*)gDirectory->Get(name);
    histx[ij] = (TH2F*)histy[ij]->Clone();
    histx[ij]->Scale(1. / nentry);
    histx[ij]->GetXaxis()->SetTitle((ityp == 0) ? "iRing" : "i#eta");
    histx[ij]->GetXaxis()->SetTitleSize(0.075);
    histx[ij]->GetXaxis()->SetTitleOffset(0.85);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.075);
    histx[ij]->GetXaxis()->SetLabelOffset(0.01);

    histx[ij]->GetYaxis()->SetTitle((ityp == 0) ? "RM" : "i#phi");
    histx[ij]->GetYaxis()->SetTitleSize(0.065);
    histx[ij]->GetYaxis()->SetTitleOffset(1.1);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.065);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    histx[ij]->GetZaxis()->SetLabelSize(0.055);
    histx[ij]->GetZaxis()->SetLabelOffset(0.01);

    histx[ij]->GetXaxis()->SetNdivisions(202);
    histx[ij]->GetYaxis()->SetNdivisions(202);
    histx[ij]->GetZaxis()->SetNdivisions(202);

    c1->cd(++icol);
    histx[ij]->Draw("colz");
    latex.DrawLatex(0.58, 0.85, Form("E_{th}=%g GeV", hothreshs[ij]));
  }
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
// plot9(2,0)
// plot9(2,1)
void plot9(int ith = 2, int ityp = 0) {
  def_setting();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.075);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right

  const int nringmx = 5;
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.07);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.20);
  gStyle->SetPadRightMargin(0.17);
  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  c1->Divide(3, 2, 1.e-6, 1.e-6);

  int nentry = ((TH2F*)gDirectory->Get("hoCalibc/hbentry_d3"))->GetBinContent(0, 0);
  TH2F* histx[6];
  TH2F* histy[6];

  sprintf(name, "hoCalibc/hormcorrel_%i", ith);

  histy[0] = (TH2F*)gDirectory->Get(name);
  for (int jk = 0; jk < nringmx; jk++) {
    if (ityp == 0) {
      sprintf(name, "hoCalibc/hocorrelsig_%i_%i", jk, ith);
    } else {
      sprintf(name, "hoCalibc/hocorrel2sig_%i_%i", jk, ith);
    }
    histy[1 + jk] = (TH2F*)gDirectory->Get(name);
  }

  for (int ij = 0; ij < 6; ij++) {
    histx[ij] = (TH2F*)histy[ij]->Clone();
    histx[ij]->Scale(1. / nentry);

    if (ij == 0) {
      histx[ij]->GetXaxis()->SetTitle("# of RM in Rings");
      histx[ij]->GetYaxis()->SetTitle("# of RM in Rings");
    } else {
      if (ityp == 0) {
        histx[ij]->GetXaxis()->SetTitle("72 #times i#eta + i#phi");
        histx[ij]->GetYaxis()->SetTitle("72 #times i#eta + i#phi");
      } else {
        sprintf(name, "%i #times i#phi + i#eta", (ij == 1 || ij == 5) ? 5 : ((ij == 3) ? 8 : 6));
        histx[ij]->GetXaxis()->SetTitle(name);
        histx[ij]->GetYaxis()->SetTitle(name);
      }
    }

    histx[ij]->GetXaxis()->SetTitleSize(0.065);
    histx[ij]->GetXaxis()->SetTitleOffset(0.90);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.065);
    histx[ij]->GetXaxis()->SetLabelOffset(0.001);

    histx[ij]->GetYaxis()->SetTitleSize(0.065);
    histx[ij]->GetYaxis()->SetTitleOffset(1.3);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.065);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    histx[ij]->GetZaxis()->SetLabelSize(0.055);
    histx[ij]->GetZaxis()->SetLabelOffset(0.01);

    histx[ij]->GetXaxis()->SetNdivisions(404);
    histx[ij]->GetYaxis()->SetNdivisions(404);
    histx[ij]->GetZaxis()->SetNdivisions(406);

    c1->cd(ij + 1);
    histx[ij]->Draw("colz");
    if (ij == 0)
      latex.DrawLatex(0.24, 0.85, Form("E_{th}=%g GeV", hothreshs[ith]));
    cout << "ij " << ij << " " << histx[ij]->GetTitle() << endl;
  }
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
//  plot10(4,5)
void plot10(int ith1 = 2, int ith2 = 3) {
  def_setting();
  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.05);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right
  const int nringmx = 5;
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetPadBottomMargin(0.10);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadRightMargin(0.16);
  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  c1->Divide(2, 1, 1.e-6, 1.e-6);
  int nentry = ((TH2F*)gDirectory->Get("hoCalibc/hbentry_d3"))->GetBinContent(0, 0);
  TH2F* histx[2];
  TH2F* histy[2];

  sprintf(name, "hoCalibc/hoallcorrelsig_%i", ith1);
  histy[0] = (TH2F*)gDirectory->Get(name);

  sprintf(name, "hoCalibc/hoallcorrelsig_%i", ith2);
  histy[1] = (TH2F*)gDirectory->Get(name);

  for (int ij = 0; ij < 2; ij++) {
    histx[ij] = (TH2F*)histy[ij]->Clone();
    histx[ij]->Scale(1. / nentry);

    histx[ij]->GetXaxis()->SetTitle("72 #times i#eta+i#phi");
    histx[ij]->GetYaxis()->SetTitle("72 #times i#eta+i#phi");

    histx[ij]->GetXaxis()->SetTitleSize(0.045);
    histx[ij]->GetXaxis()->SetTitleOffset(0.90);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.045);
    histx[ij]->GetXaxis()->SetLabelOffset(0.001);

    histx[ij]->GetYaxis()->SetTitleSize(0.045);
    histx[ij]->GetYaxis()->SetTitleOffset(1.65);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.045);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    histx[ij]->GetZaxis()->SetLabelSize(0.045);
    histx[ij]->GetZaxis()->SetLabelOffset(0.01);

    histx[ij]->GetXaxis()->SetNdivisions(406);
    histx[ij]->GetYaxis()->SetNdivisions(406);
    histx[ij]->GetZaxis()->SetNdivisions(406);
    if (ij == 0)
      histx[ij]->SetMaximum(0.000001);
    //    if (ij==1) histx[ij]->SetMaximum(0.0000005);
    c1->cd(ij + 1);
    histx[ij]->Draw("colz");
    latex.DrawLatex(0.28, 0.85, Form("E_{th}=%g GeV", hothreshs[(ij == 0) ? ith1 : ith2]));

    cout << "ij " << histx[ij]->GetTitle() << endl;
  }
  c1->Update();
}

// root histall_apr14b_cosmic_csa14_cosmic.root
//NO  plot11(2,3)
void plot11(int ith1 = 2, int ith2 = 3) {
  def_setting();

  TLatex latex;
  latex.SetNDC();
  latex.SetTextSize(0.05);
  latex.SetTextFont(42);
  latex.SetTextAlign(1);  //(31); // align right
  const int nringmx = 5;
  gStyle->SetOptStat(0);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetPadBottomMargin(0.10);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadRightMargin(0.16);

  char name[100];
  TCanvas* c1 = new TCanvas("c1", "c1", 800., 500.);
  c1->Divide(2, 1, 1.e-6, 1.e-6);
  int nentry = ((TH2F*)gDirectory->Get("hoCalibc/hbentry_d3"))->GetBinContent(0, 0);
  TH2F* histx[2];
  TH2F* histy[2];

  sprintf(name, "hoCalibc/hoallmucorrel_%i", ith1);
  histy[0] = (TH2F*)gDirectory->Get(name);

  sprintf(name, "hoCalibc/hoallmucorrel_%i", ith2);
  histy[1] = (TH2F*)gDirectory->Get(name);

  for (int ij = 0; ij < 2; ij++) {
    double total = 0;
    histx[ij] = (TH2F*)histy[ij]->Clone();
    for (int ix = 1; ix <= histy[ij]->GetNbinsX(); ix++) {
      float anent = histy[ij]->GetBinContent(ix, 0);
      total += anent;
      //      cout<<"ix "<< ij<<" "<<ix<<" "<<anent<<" "<<total<<endl;
      if (anent < 1.)
        anent = 1.;
      for (int iy = 1; iy <= histy[ij]->GetNbinsY(); iy++) {
        histx[ij]->SetBinContent(ix, iy, histy[ij]->GetBinContent(ix, iy) / anent);
      }
    }

    histx[ij]->GetXaxis()->SetTitle("Projected muon (72 #times i#eta+i#phi)");
    histx[ij]->GetYaxis()->SetTitle("Signal in HO tower (72 #times i#eta+i#phi)");

    histx[ij]->GetXaxis()->SetTitleSize(0.045);
    histx[ij]->GetXaxis()->SetTitleOffset(0.90);  //0.85
    histx[ij]->GetXaxis()->CenterTitle();
    histx[ij]->GetXaxis()->SetLabelSize(0.04);
    histx[ij]->GetXaxis()->SetLabelOffset(0.001);

    histx[ij]->GetYaxis()->SetTitleSize(0.045);
    histx[ij]->GetYaxis()->SetTitleOffset(1.65);
    histx[ij]->GetYaxis()->CenterTitle();
    histx[ij]->GetYaxis()->SetLabelSize(0.04);
    histx[ij]->GetYaxis()->SetLabelOffset(0.01);

    histx[ij]->GetZaxis()->SetLabelSize(0.04);
    histx[ij]->GetZaxis()->SetLabelOffset(0.01);

    histx[ij]->GetXaxis()->SetNdivisions(406);
    histx[ij]->GetYaxis()->SetNdivisions(406);
    histx[ij]->GetZaxis()->SetNdivisions(406);
    //    if (ij==0) histx[ij]->SetMaximum(0.002);
    //    if (ij==1) histx[ij]->SetMaximum(0.001);
    c1->cd(ij + 1);
    histx[ij]->Draw("colz");
    latex.DrawLatex(0.28, 0.85, Form("E_{th}=%g GeV", hothreshs[(ij == 0) ? ith1 : ith2]));
    cout << "ij " << histx[ij]->GetTitle() << endl;
  }
  c1->Update();
}

/*

root ../local/loc2014/hist_local_*_Cosmics.root
 .L anal_csa14_pl.C
alllocal(6);
root ../local/loc2014/hist_local_*_40fC.root
 .L anal_csa14_pl.C
alllocal(6);
root ../local/loc2014/hist_local_*_50fC.root
 .L anal_csa14_pl.C
alllocal(3);
root ../local/loc2014/hist_local_*_60fC.root
 .L anal_csa14_pl.C
alllocal(9);
root ../local/loc2014/hist_local_*_peds.root
 .L anal_csa14_pl.C
alllocal(3);
*/

void alllocal(int nfile = 9) {
  char pch1[200];
  for (int ij = 0; ij < nfile; ij++) {
    switch (ij) {
      case 0:
        _file0->cd();
        break;
      case 1:
        _file1->cd();
        break;
      case 2:
        _file2->cd();
        break;
      case 3:
        _file3->cd();
        break;
      case 4:
        _file4->cd();
        break;
      case 5:
        _file5->cd();
        break;
      case 6:
        _file6->cd();
        break;
      case 7:
        _file7->cd();
        break;
      case 8:
        _file8->cd();
        break;
      case 9:
        _file9->cd();
        break;
      default:
        _file0->cd();
        break;
    }

    cout << "xx1 " << endl;
    char* namex = gDirectory->GetName();
    int len2 = strlen(namex);
    strncpy(pch1, namex + 17, len2 - 22);
    pch1[len2 - 22] = '\0';
    sprintf(pch1, "%s.ps", pch1);
    TPostScript pss(pch1, 112);
    pss.Range(28, 20);  //pss.Range(20,28);
    cout << "xx2 " << endl;
    pss.NewPage();
    pss.NewPage();
    testxx();
    pss.NewPage();
    /*
    pss.NewPage(); plot000(-1,6,6,1); 
    pss.NewPage(); plot000(0,6,6,1);
    pss.NewPage(); plot000(1,6,6,1); 
    pss.NewPage(); plot000(2,12,12,1);
    pss.NewPage(); plot000(3,6,6,1); 
    pss.NewPage(); plot000(4,6,6,1);
    */

    pss.NewPage();
    plot000(0, 6, 6, 1);
    plot000(1, 6, 6, 1, "same", 1);
    plot000(2, 12, 12, 1, "same", 2);
    plot000(3, 6, 6, 1, "same", 3);
    plot000(4, 6, 6, 1, "sames", 4);

    for (int jk = 0; jk < 5; jk++) {
      pss.NewPage();
      plot000(jk, 0, 0, 1);
      for (int kl = 1; kl < 12; kl++) {
        if (jk != 2 && kl > 5)
          continue;
        cout << "jk " << jk << " " << kl << endl;
        if ((jk == 2 && kl == 11) || (jk != 2 && kl == 5)) {
          plot000(jk, kl, kl, 1, "sames", kl);
        } else {
          plot000(jk, kl, kl, 1, "same", kl);
        }
      }
    }

    //     //    pss.NewPage(); plot000(1,0,0,1);
    //     //    plot000(1,1,1,1, "same",1);
    //     //    plot000(1,2,2,1, "same",2);
    //     //    plot000(1,3,3,1, "same",3);
    //     //    plot000(1,4,4,1, "same",4);
    //     //    plot000(1,5,5,1, "sames",5);

    // //     /*
    // //     pss.NewPage(); plot000(0,0,5,1);
    // //     pss.NewPage(); plot000(1,0,5,1);
    // //     pss.NewPage(); plot000(2,0,5,1);
    // //     pss.NewPage(); plot000(2,6,11,1);
    // //     pss.NewPage(); plot000(3,0,5,1);
    // //     pss.NewPage(); plot000(4,0,5,1);
    // //     */

    pss.NewPage();
    plot000r(-1);
    pss.NewPage();
    plot000r(0);
    pss.NewPage();
    plot000r(1);
    pss.NewPage();
    plot000r(2);
    pss.NewPage();
    plot000r(3);
    pss.NewPage();
    plot000r(4);

    // //     /*
    // //     pss.NewPage(); plot01(0,1);
    // //     pss.NewPage(); plot01(1,1);
    // //     */
    pss.NewPage();
    plot01(2, 1);

    pss.NewPage();
    plot2(0, 4, 0);
    pss.NewPage();
    plot2(4, 8, 0);

    // //     pss.NewPage(); plot4(0,8,0,0);
    // //     //    pss.NewPage(); plot4(3,5,0,0);
    // //     //    pss.NewPage(); plot4(6,8,0,0);

    // //     //    pss.NewPage(); plot5(0,2,0,0);
    // //     //    pss.NewPage(); plot5(3,5,0,0);
    // //     //    pss.NewPage(); plot5(6,8,0,0);

    pss.NewPage();
    plot6(-2, 0);
    pss.NewPage();
    plot6(-1, 0);
    pss.NewPage();
    plot6(0, 0);
    pss.NewPage();
    plot6(1, 0);
    pss.NewPage();
    plot6(2, 0);

    // // //     pss.NewPage(); plot7(0,0);
    // // //     pss.NewPage(); plot7(1,0);
    // // //     pss.NewPage(); plot7(2,0);
    // // //     pss.NewPage(); plot7(3,0);
    // // //     pss.NewPage(); plot7(4,0);
    // // //     pss.NewPage(); plot7(2,2);

    pss.NewPage();
    plot7b(-1, 3);
    pss.NewPage();
    plot7b(0, 2, 8, 0);
    pss.NewPage();
    plot7b(1, 2, 8, 0);
    pss.NewPage();
    plot7b(2, 2, 8, 0);
    pss.NewPage();
    plot7b(3, 2, 8, 0);
    pss.NewPage();
    plot7b(4, 2, 8, 0);
    pss.NewPage();
    plot7b(0, 2, 8, 1);
    pss.NewPage();
    plot7b(1, 2, 8, 1);
    pss.NewPage();
    plot7b(2, 2, 8, 1);
    pss.NewPage();
    plot7b(3, 2, 8, 1);
    pss.NewPage();
    plot7b(4, 2, 8, 1);

    pss.NewPage();
    plot8(1, 8, 0);
    pss.NewPage();
    plot8(1, 8, 1);

    pss.NewPage();
    plot9(2, 0);
    pss.NewPage();
    plot9(2, 1);

    //    pss.NewPage(); plot10(4,5);

    pss.Close();
  }
}
/*
hadd histall_local_peds.root hist_local_*_peds.root
hadd histall_local_40fc.root hist_local_22*_40fC.root
hadd histall_local_50fc.root hist_local_22*_50fC.root
hadd histall_local_60fc.root hist_local_221848_60fC.root hist_local_221945_60fC.root hist_local_222770_60fC.root
hadd histall_local_cosmics.root hist_local_220619_Cosmics.root hist_local_220620_Cosmics.root hist_local_220625_Cosmics.root
*/

void alllocalx(int nfile = 9) {
  char pch1[200];
  for (int ij = 0; ij < nfile; ij++) {
    switch (ij) {
      case 0:
        _file0->cd();
        break;
      case 1:
        _file1->cd();
        break;
      case 2:
        _file2->cd();
        break;
      case 3:
        _file3->cd();
        break;
      case 4:
        _file4->cd();
        break;
      case 5:
        _file5->cd();
        break;
      case 6:
        _file6->cd();
        break;
      case 7:
        _file7->cd();
        break;
      case 8:
        _file8->cd();
        break;
      case 9:
        _file9->cd();
        break;
      case 10:
        _file10->cd();
        break;
      case 11:
        _file11->cd();
        break;
      case 12:
        _file12->cd();
        break;
      case 13:
        _file13->cd();
        break;
      case 14:
        _file14->cd();
        break;
      case 15:
        _file15->cd();
        break;
      case 16:
        _file16->cd();
        break;
      case 17:
        _file17->cd();
        break;
      case 18:
        _file18->cd();
        break;
      case 19:
        _file19->cd();
        break;
      case 20:
        _file20->cd();
        break;
      case 21:
        _file21->cd();
        break;
      case 22:
        _file22->cd();
        break;
      case 23:
        _file23->cd();
        break;
      case 24:
        _file24->cd();
        break;
      case 25:
        _file25->cd();
        break;
      case 26:
        _file26->cd();
        break;
      case 27:
        _file27->cd();
        break;
      case 28:
        _file28->cd();
        break;
      case 29:
        _file29->cd();
        break;

      default:
        _file0->cd();
        break;
    }

    char* namex = gDirectory->GetName();
    int len2 = strlen(namex);
    strncpy(pch1, namex + 17, len2 - 22);
    pch1[len2 - 22] = '\0';
    sprintf(pch1, "xx_%s.ps", pch1);
    TPostScript pss(pch1, 112);
    pss.Range(28, 20);  //pss.Range(20,28);

    pss.NewPage();
    pss.NewPage();
    testxx();
    pss.NewPage();
    testxx();

    pss.NewPage();
    plot000r(-1, 0, 140);
    pss.NewPage();
    plot000r(-1, 1, 140);

    //    pss.NewPage(); plot000r(0);
    //    pss.NewPage(); plot000r(1);
    //    pss.NewPage(); plot000r(2);
    //    pss.NewPage(); plot000r(3);
    //    pss.NewPage(); plot000r(4);

    pss.NewPage();
    plot2(0, 4, 0);
    pss.NewPage();
    plot2(4, 8, 0);

    //    pss.NewPage(); plot000r(0);
    //    pss.NewPage(); plot000r(1);
    //    pss.NewPage(); plot000r(2);
    //    pss.NewPage(); plot000r(3);
    //    pss.NewPage(); plot000r(4);

    pss.NewPage();
    plot01(2, 1);

    pss.NewPage();
    plot7b(-1, 2, 8);
    pss.NewPage();
    plot7b(0, 2, 8, 0);
    pss.NewPage();
    plot7b(1, 2, 8, 0);
    pss.NewPage();
    plot7b(2, 2, 8, 0);
    pss.NewPage();
    plot7b(3, 2, 8, 0);
    pss.NewPage();
    plot7b(4, 2, 8, 0);
    pss.NewPage();
    plot7b(0, 2, 8, 1);
    pss.NewPage();
    plot7b(1, 2, 8, 1);
    pss.NewPage();
    plot7b(2, 2, 8, 1);
    pss.NewPage();
    plot7b(3, 2, 8, 1);
    pss.NewPage();
    plot7b(4, 2, 8, 1);

    pss.NewPage();
    plot8(1, 8, 0);
    pss.NewPage();
    plot8(1, 8, 1);

    pss.NewPage();
    plot9(2, 0);
    //     pss.NewPage(); plot9(2,1);

    //    pss.NewPage(); plot10(4,5);

    pss.NewPage();
    plot4(0, 8, 0, 0);

    // //     //    pss.NewPage(); plot5(0,2,0,0);
    // //     //    pss.NewPage(); plot5(3,5,0,0);
    // //     //    pss.NewPage(); plot5(6,8,0,0);

    pss.NewPage();
    plot000(0, 6, 6, 1);
    plot000(1, 6, 6, 1, "same", 1);
    plot000(2, 12, 12, 1, "same", 2);
    plot000(3, 6, 6, 1, "same", 3);
    plot000(4, 6, 6, 1, "sames", 4);

    for (int jk = 0; jk < 5; jk++) {
      pss.NewPage();
      plot000(jk, 0, 0, 1);
      for (int kl = 1; kl < 12; kl++) {
        if (jk != 2 && kl > 5)
          continue;
        cout << "jk " << jk << " " << kl << endl;
        if ((jk == 2 && kl == 11) || (jk != 2 && kl == 5)) {
          plot000(jk, kl, kl, 1, "sames", kl);
        } else {
          plot000(jk, kl, kl, 1, "same", kl);
        }
      }
    }

    pss.NewPage();
    plot6(-2, 0);
    pss.NewPage();
    plot6(-1, 0);
    pss.NewPage();
    plot6(0, 0);
    pss.NewPage();
    plot6(1, 0);
    pss.NewPage();
    plot6(2, 0);

    pss.Close();
  }
}

// scp hcal_local_*_peds.root hcal_local_*_40fC.root hcal_local_*_50fC.root hcal_local_*_60fC.root hist_local_*_peds.root hist_local_*_40fC.root hist_local_*_50fC.root hist_local_*_60fC.root gobinda@158.144.54.116:/data/gobinda/anal/hcal/local/loc2014/

void test1x() {
  for (int ij = 1; ij < 14; ij++) {
    cout << "INR " << 6200 * ij + 4000 << " (for " << ij << " nights)" << endl;
  }
}

void sigpedrun(int nrn = 100) {
  TPostScript ps("testxx.ps", 111);
  ps.Range(20, 28);
  TCanvas* c0x = new TCanvas("c0x", " Pedestal and signal", 900, 1200);
  c0x->Divide(5, 4, 1.e-5, 1.e-5, 0);

  TH1F* signal_run[5][3000];
  char name[100];

  for (int ix = 0; ix < nrn; ix++) {
    int ixxy = ix % 4;
    if (ixxy == 0) {
      ps.NewPage();
    }
    for (int iy = 0; iy < ringmx; iy++) {
      sprintf(name, "noise_ring_%i_run%i", iy, ix);
      signal_run[iy][ix] = (TH1F*)gDirectory->Get(name);
      double mean = signal_run[iy][ix]->GetMean();
      double rms = signal_run[iy][ix]->GetRMS();
      if (iy == 0)
        cout << ix << " " << signal_run[iy][ix]->GetTitle() << " ";
      //      cout <<" "<<signal_run[iy][ix]->GetEntries()<<" "<<mean<<" "<<rms;
      cout << " " << mean;
      c0x->cd(ringmx * ixxy + iy + 1);
      signal_run[iy][ix]->SetLineColor(3);
      signal_run[iy][ix]->GetXaxis()->SetLabelSize(0.095);
      signal_run[iy][ix]->GetXaxis()->SetNdivisions(404);
      signal_run[iy][ix]->GetYaxis()->SetLabelSize(0.095);
      signal_run[iy][ix]->GetXaxis()->SetLabelOffset(-0.02);
      signal_run[iy][ix]->Draw();
      if (ixxy == 3) {
        c0x->Update();
      }
    }
    cout << endl;
  }

  ps.Close();
}
/*
 Run # 297180 Evt # 102392070 1 7024424

All    94.67     92.49     95.35      92.7     94.71 

ndof    93.37        92     94.58     92.21      93.3 
chisq    93.34     91.96     94.53     92.16     93.27 
angle    93.33     91.96     94.53     92.16     93.27 
pt    92.13     90.45     92.22     90.63     92.01 
isol    62.28        56     55.51     56.17     62.68 
phi    56.27      50.7     47.38     50.85     56.58 
eta    52.41     46.51     36.64     46.69      52.7 
Time    52.24      46.4     36.63     46.54     52.54 


hodx    99.31     97.74     99.62     98.03     99.32 
iso    96.61     94.44        97     94.67     96.68 
pt     96.4     94.17     96.81      94.4     96.48 



   1  Constant     8.56917e+01   2.42013e+00   7.85930e-03   3.06102e-06
   2  Mean         1.00342e+00   1.07020e-03   4.43892e-06   3.20972e-03
   3  Sigma        4.79270e-02   8.60831e-04   1.86192e-05  -5.88994e-03

   1  Constant     1.92784e+02   5.34817e+00   1.63458e-02  -1.85787e-06
   2  Mean         9.99574e-01   4.75653e-04   1.82064e-06   2.98504e-02
   3  Sigma        2.14641e-02   3.63935e-04   1.51944e-05  -3.40005e-03

   1  Constant     4.23168e+02   1.23598e+01   3.79784e-02  -4.19190e-05
   2  Mean         1.01199e+00   2.17199e-04   8.81854e-07  -3.90635e-01
   3  Sigma        9.80276e-03   1.89893e-04   1.59833e-05  -2.89784e-02

   1  Constant     2.73388e+02   7.95311e+00   2.29623e-02   1.00544e-07
   2  Mean         1.01965e+00   3.34716e-04   1.28072e-06   1.32401e-01
   3  Sigma        1.52308e-02   2.92214e-04   1.60562e-05  -4.93738e-03

   1  Constant     3.33148e+02   9.67580e+00   2.88215e-02  -1.45420e-05
   2  Mean         1.02102e+00   2.74804e-04   1.08052e-06  -2.43768e+00
   3  Sigma        1.24753e-02   2.37824e-04   1.60597e-05   3.78574e-02

   1  Constant     4.11514e+02   1.19379e+01   3.34777e-02  -1.48396e-07
   2  Mean         1.02247e+00   2.23653e-04   8.24581e-07   3.21928e-03
   3  Sigma        1.01355e-02   1.93182e-04   1.45150e-05  -2.31771e-04

   1  Constant     4.17155e+02   1.22384e+01   3.67076e-02  -3.41070e-05
   2  Mean         1.02334e+00   2.19820e-04   8.75434e-07  -4.30174e+00
   3  Sigma        9.95279e-03   1.94035e-04   1.57779e-05   1.98571e-02

   1  Constant     3.99846e+02   1.20076e+01   4.10521e-02   1.64058e-06
   2  Mean         1.02511e+00   2.34554e-04   1.05387e-06   4.74777e-02
   3  Sigma        1.02668e-02   2.09319e-04   1.83947e-05   1.76225e-02


CRO_pulse->Draw("c2PeakTime_1[0]","c1PeakTime_1[0]>-100&&c2PeakTime_1[0]>-100&&")

CRO_pulse->Draw("c4PeakTime_1[1]","c3PeakTime_1[1]>-100&&c4PeakTime_1[1]>-100&&c3PeakTime_1[1]>1.e-6&&c4PeakTime_1[1]>1.e-6")


*/
