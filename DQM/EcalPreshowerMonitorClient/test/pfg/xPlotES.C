void xPlotES(Int_t run) {

  gROOT->Reset();

  Char_t fname[200];
  sprintf(fname, "DQM_V0001_EcalPreshower_%06d.root", run);

  TFile *f = new TFile(fname);

  TCanvas *c1 = new TCanvas("c1", "c1", 500, 500); 

  Char_t hname[600];
  Char_t tname[600];

  // ES Occupancy
  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 1", run);
  sprintf(tname, "ES Occupancy ES+F");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_2D_ESpF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES RecHit 2D Occupancy Z 1 P 2", run);
  sprintf(tname, "ES Occupancy ES+R");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_2D_ESpR_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 1", run);
  sprintf(tname, "ES Occupancy ES-F");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_2D_ESmF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES RecHit 2D Occupancy Z -1 P 2", run);
  sprintf(tname, "ES Occupancy ES-R");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_2D_ESmR_%06d.gif", run);
  c1->Print(hname);

  // ES Energy Density
  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density Z 1 P 1", run);
  sprintf(tname, "ES Energy Density ES+F");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_2D_ESpF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density Z 1 P 2", run);
  sprintf(tname, "ES Energy Density ES+R");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_2D_ESpR_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density Z -1 P 1", run);
  sprintf(tname, "ES Energy Density ES-F");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_2D_ESmF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density Z -1 P 2", run);
  sprintf(tname, "ES Energy Density ES-R");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_2D_ESmR_%06d.gif", run);
  c1->Print(hname);

  // ES Occupancy with selected hits
  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Occupancy with selected hits Z 1 P 1", run);
  sprintf(tname, "ES Occupancy with selected hits ES+F");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_SelHits_2D_ESpF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Occupancy with selected hits Z 1 P 2", run);
  sprintf(tname, "ES Occupancy with selected hits ES+R");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_SelHits_2D_ESpR_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Occupancy with selected hits Z -1 P 1", run);
  sprintf(tname, "ES Occupancy with selected hits ES-F");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_SelHits_2D_ESmF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Occupancy with selected hits Z -1 P 2", run);
  sprintf(tname, "ES Occupancy with selected hits ES-R");
  draw2dOcc(f, hname, tname, c1);
  sprintf(hname, "ES_Occupancy_SelHits_2D_ESmR_%06d.gif", run);
  c1->Print(hname);

  // ES Energy Density
  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density with selected hits Z 1 P 1", run);
  sprintf(tname, "ES Energy Density with selected hits ES+F");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_SelHits_2D_ESpF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density with selected hits Z 1 P 2", run);
  sprintf(tname, "ES Energy Density with selected hits ES+R");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_SelHits_2D_ESpR_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density with selected hits Z -1 P 1", run);
  sprintf(tname, "ES Energy Density with selected hits ES-F");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_SelHits_2D_ESmF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Energy Density with selected hits Z -1 P 2", run);
  sprintf(tname, "ES Energy Density with selected hits ES-R");
  draw2dEn(f, hname, tname, c1);
  sprintf(hname, "ES_EnergyDensity_SelHits_2D_ESmR_%06d.gif", run);
  c1->Print(hname);

  // ES Occupancy 1D
  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of RecHits Z 1 P 1", run);
  sprintf(tname, "ES Number of rec hits ES+F");
  draw1d(f, hname, tname, c1, 4, 1, 1000);
  sprintf(hname, "ES_Occupancy_1D_ESpF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of RecHits Z 1 P 2", run);
  sprintf(tname, "ES Number of rec hits ES+R");
  draw1d(f, hname, tname, c1, 4, 1, 1000);
  sprintf(hname, "ES_Occupancy_1D_ESpR_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of RecHits Z -1 P 1", run);
  sprintf(tname, "ES Number of rec hits ES-F");
  draw1d(f, hname, tname, c1, 4, 1, 1000);
  sprintf(hname, "ES_Occupancy_1D_ESmF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of RecHits Z -1 P 2", run);
  sprintf(tname, "ES Number of rec hits ES-R");
  draw1d(f, hname, tname, c1, 4, 1, 1000);
  sprintf(hname, "ES_Occupancy_1D_ESmR_%06d.gif", run);
  c1->Print(hname);

  // ES Occupancy 1D with selected hits
  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of Good RecHits Z 1 P 1", run);
  sprintf(tname, "ES Number of good rec hits ES+F");
  draw1d(f, hname, tname, c1, 2, 1, 1000);
  sprintf(hname, "ES_Occupancy_SelHits_1D_ESpF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of Good RecHits Z 1 P 2", run);
  sprintf(tname, "ES Number of good rec hits ES+R");
  draw1d(f, hname, tname, c1, 2, 1, 1000);
  sprintf(hname, "ES_Occupancy_SelHits_1D_ESpR_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of Good RecHits Z -1 P 1", run);
  sprintf(tname, "ES Number of good rec hits ES-F");
  draw1d(f, hname, tname, c1, 2, 1, 1000);
  sprintf(hname, "ES_Occupancy_SelHits_1D_ESmF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESOccupancyTask/ES Num of Good RecHits Z -1 P 2", run);
  sprintf(tname, "ES Number of good rec hits ES-R");
  draw1d(f, hname, tname, c1, 2, 1, 1000);
  sprintf(hname, "ES_Occupancy_SelHits_1D_ESmR_%06d.gif", run);
  c1->Print(hname);

  // ES timing plots
  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESTimingTask/ES 2D Timing", run);
  draw2d(f, hname, c1);
  sprintf(hname, "ES_Timing_2D_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESTimingTask/ES Timing Z 1 P 1", run);
  sprintf(tname, "ES Timing ES+F");
  draw1dFitGaus(f, hname, tname, c1);
  sprintf(hname, "ES_Timing_ESpF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESTimingTask/ES Timing Z 1 P 2", run);
  sprintf(tname, "ES Timing ES+R");
  draw1dFitGaus(f, hname, tname, c1);
  sprintf(hname, "ES_Timing_ESpR_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESTimingTask/ES Timing Z -1 P 1", run);
  sprintf(tname, "ES Timing ES-F");
  draw1dFitGaus(f, hname, tname, c1);
  sprintf(hname, "ES_Timing_ESmF_%06d.gif", run);
  c1->Print(hname);

  c1->Clear();
  sprintf(hname, "DQMData/Run %06d/EcalPreshower/Run summary/ESTimingTask/ES Timing Z -1 P 2", run);
  sprintf(tname, "ES Timing ES-R");
  draw1dFitGaus(f, hname, tname, c1);
  sprintf(hname, "ES_Timing_ESmR_%06d.gif", run);
  c1->Print(hname);
}  

void draw1dFitGaus(TFile *f, Char_t *hname, Char_t *tname, TCanvas *c) {

  TH1F *h1 = (TH1F*) f->Get(hname);
  TF1 *g = new TF1("g", "gaus");
  h1->Fit("g", "Q");

  //gStyle->SetOptStat(10);
  c->cd();
  gPad->SetLogy(0);
  h1->SetTitle(tname);
  h1->Draw();

  return;
}

void draw1d(TFile *f, Char_t *hname, Char_t *tname, TCanvas *c, Int_t color=1, Int_t log=0, Int_t Range=1000) {

  TH1F *h1 = (TH1F*) f->Get(hname);

  gStyle->SetOptStat(1110);
  c->cd();
  gPad->SetLogy(log);
  h1->SetTitle(tname);
  h1->SetLineColor(color);
  h1->GetXaxis()->SetLimits(0, Range);
  h1->Draw();

  return;
}

void draw2d(TFile *f, Char_t *hname, TCanvas *c) {

  TH2F *h2 = (TH2F*) f->Get(hname);

  gStyle->SetOptStat(10);
  c->cd();
  gPad->SetLogy(0);
  h2->Draw("colz");

  return;
}

void draw2dOcc(TFile *f, Char_t *hname, Char_t *tname, TCanvas *c) {

  TH2F *h2 = (TH2F*) f->Get(hname);

  float NEntries = (float) h2->GetBinContent(40,40);
  h2->SetBinContent(40,40,0.);
  h2->Scale(1/NEntries);
  h2->SetMaximum(32);

  gStyle->SetOptStat(0);
  c->cd();
  gPad->SetLogy(0);
  h2->SetTitle(tname);
  h2->Draw("colz");

  return;
}

void draw2dEn(TFile *f, Char_t *hname, Char_t *tname, TCanvas *c) {

  TH2F *h2 = (TH2F*) f->Get(hname);

  float NEntries = (float) h2->GetBinContent(40,40);
  h2->SetBinContent(40,40,0.);
  h2->Scale(1/NEntries);

  gStyle->SetOptStat(0);
  c->cd();
  gPad->SetLogy(0);
  h2->SetTitle(tname);
  h2->Draw("colz");

  return;
}

