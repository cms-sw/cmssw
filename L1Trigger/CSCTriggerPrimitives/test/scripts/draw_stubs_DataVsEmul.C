/*
 A script to draw several data vs. emulator comparison histograms.
The input histograms are taken from a root file produced by the CSCTriggerPrimitivesReader module.

To run, e.g.:

.x draw_stubs_DataVsEmul.C("TPEHists.root")

*/


TH2F *hDataVsEmuME11_nALCTPerCSC, *hDataVsEmuME11_nCLCTPerCSC, *hDataVsEmuME11_nLCTPerCSC;
TH2F *hDataVsEmuME11_ALCTWG, *hDataVsEmuME11_CLCTHS, *hDataVsEmuME11_LCTWG, *hDataVsEmuME11_LCTHS;
TH2F *hDataVsEmuME11_ALCTQ, *hDataVsEmuME11_CLCTQ, *hDataVsEmuME11_LCTQ;
TH2F *hDataME11_LCTHSvsWG, *hEmuME11_LCTHSvsWG;


void setupH(TH2F*h, int ndiv = 109)
{
h->GetXaxis()->SetNdivisions(ndiv);
h->GetYaxis()->SetNdivisions(ndiv);
h->GetZaxis()->SetLabelSize(0.04);
h->GetZaxis()->SetLabelSize(0.04);
}


TH2F* deltaH(TH2F*h1, TH2F*h2)
{
TH2F*h = (TH2F*)h1->Clone(Form("%s_delta",h1->GetName()));
float dmax = 0.;
for (int i=1; i<=h->GetNbinsX(); ++i)
  for (int j=1; j<=h->GetNbinsY(); ++j)
  {
    int b1 = h1->GetBinContent(i,j);
    int b2 = h2->GetBinContent(i,j);
    if (b1==0 && b2==0) { h->SetBinContent(i,j,-1000000); continue; }
    h->SetBinContent(i,j,b1-b2);
    h->SetBinError(i,j,1);
    if (dmax < fabs(b1-b2)) dmax = fabs(b1-b2);
  }
if (dmax < 1) dmax = 1;
h->GetZaxis()->SetRangeUser(-dmax,dmax);
return h;
}


void draw_stubs_DataVsEmul(TString histo_file_name = "TPEHists.root")
{
gROOT->SetBatch(true);

//gStyle->SetTitleSize(0.001);
gStyle->SetOptStat(0);

TFile *f = TFile::Open(histo_file_name);

hDataVsEmuME11_nALCTPerCSC = (TH2F*) f->Get("lctreader/hDataVsEmuME11_nALCTPerCSC");
hDataVsEmuME11_nCLCTPerCSC = (TH2F*) f->Get("lctreader/hDataVsEmuME11_nCLCTPerCSC");
hDataVsEmuME11_nLCTPerCSC = (TH2F*) f->Get("lctreader/hDataVsEmuME11_nLCTPerCSC");
hDataVsEmuME11_ALCTWG = (TH2F*) f->Get("lctreader/hDataVsEmuME11_ALCTWG");
hDataVsEmuME11_CLCTHS = (TH2F*) f->Get("lctreader/hDataVsEmuME11_CLCTHS");
hDataVsEmuME11_LCTWG = (TH2F*) f->Get("lctreader/hDataVsEmuME11_LCTWG");
hDataVsEmuME11_LCTHS = (TH2F*) f->Get("lctreader/hDataVsEmuME11_LCTHS");
hDataVsEmuME11_ALCTQ = (TH2F*) f->Get("lctreader/hDataVsEmuME11_ALCTQ");
hDataVsEmuME11_CLCTQ = (TH2F*) f->Get("lctreader/hDataVsEmuME11_CLCTQ");
hDataVsEmuME11_LCTQ = (TH2F*) f->Get("lctreader/hDataVsEmuME11_LCTQ");
hDataME11_LCTHSvsWG = (TH2F*) f->Get("lctreader/hDataME11_LCTHSvsWG");
hEmuME11_LCTHSvsWG = (TH2F*) f->Get("lctreader/hEmuME11_LCTHSvsWG");

setupH(hDataVsEmuME11_nALCTPerCSC);
setupH(hDataVsEmuME11_nCLCTPerCSC);
setupH(hDataVsEmuME11_nLCTPerCSC);
setupH(hDataVsEmuME11_ALCTWG, 510);
setupH(hDataVsEmuME11_CLCTHS, 510);
setupH(hDataVsEmuME11_LCTWG, 510);
setupH(hDataVsEmuME11_LCTHS, 510);
setupH(hDataVsEmuME11_ALCTQ);
setupH(hDataVsEmuME11_CLCTQ);
setupH(hDataVsEmuME11_LCTQ, 117);
setupH(hDataME11_LCTHSvsWG, 113);
setupH(hEmuME11_LCTHSvsWG, 113);

TCanvas *c = new TCanvas("c1","c1",800,750);
gPad->SetRightMargin(0.14);
gPad->SetGridx(1);
gPad->SetGridy(1);

hDataVsEmuME11_nALCTPerCSC->Draw("hcolz text");
c->Print("hDataVsEmuME11_nALCTPerCSC.png");
hDataVsEmuME11_nCLCTPerCSC->Draw("hcolz");
c->Print("hDataVsEmuME11_nCLCTPerCSC.png");
hDataVsEmuME11_nLCTPerCSC->Draw("hcolz");
c->Print("hDataVsEmuME11_nLCTPerCSC.png");
hDataVsEmuME11_ALCTWG->Draw("hcolz");
c->Print("hDataVsEmuME11_ALCTWG.png");
hDataVsEmuME11_CLCTHS->Draw("hcolz");
c->Print("hDataVsEmuME11_CLCTHS.png");
hDataVsEmuME11_LCTWG->Draw("hcolz");
c->Print("hDataVsEmuME11_LCTWG.png");
hDataVsEmuME11_LCTHS->Draw("hcolz");
c->Print("hDataVsEmuME11_LCTHS.png");
hDataVsEmuME11_ALCTQ->Draw("hcolz");
c->Print("hDataVsEmuME11_ALCTQ.png");
hDataVsEmuME11_CLCTQ->Draw("hcolz");
c->Print("hDataVsEmuME11_CLCTQ.png");
hDataVsEmuME11_LCTQ->Draw("hcolz");
c->Print("hDataVsEmuME11_LCTQ.png");
gPad->SetLogz(1);
hDataME11_LCTHSvsWG->Draw("hcolz");
c->Print("hDataME11_LCTHSvsWG.png");
hEmuME11_LCTHSvsWG->SetXTitle("Half-strip");
hEmuME11_LCTHSvsWG->SetYTitle("Wire-group");
hEmuME11_LCTHSvsWG->Draw("hcolz");
c->Print("hEmuME11_LCTHSvsWG.png");

gPad->SetLogz(0);
TH2F* d = deltaH(hDataME11_LCTHSvsWG, hEmuME11_LCTHSvsWG);
d->SetTitle("Difference (Data - Emulator): LCT HS vs WG in ME1/1");
d->Draw("hcolz");
c->Print("hDeltaDataEmuME11_LCTHSvsWG.png");

}
