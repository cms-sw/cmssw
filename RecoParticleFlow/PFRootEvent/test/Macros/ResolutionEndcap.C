{
//author A.Zabi
//This macro can be used to reproduce the tau benchmark plot
//for tau jet reconstruction studies
// 50 GeV taus desintegrating hadronically have been studied
gROOT->Reset();
TFile *f = new TFile("pfjetBenchmark.root");

//ERPt20_40->Fit("gaus","L");
//ERPt40_60->Fit("gaus","L");
//ERPt60_80->Fit("gaus","L");
//ERPt80_100->Fit("gaus","L");
//ERPt100_150->Fit("gaus","L");
//ERPt150_200->Fit("gaus","L");
//ERPt200_250->Fit("gaus","L");
//ERPt250_300->Fit("gaus","L");
//ERPt300_400->Fit("gaus","L");
//ERPt400_500->Fit("gaus","L");
//ERPt500_750->Fit("gaus","L");

ERPt20_40->Fit("gaus");
ERPt40_60->Fit("gaus");
ERPt60_80->Fit("gaus");
ERPt80_100->Fit("gaus");
ERPt100_150->Fit("gaus");
ERPt150_200->Fit("gaus");
ERPt200_250->Fit("gaus");
ERPt250_300->Fit("gaus");
ERPt300_400->Fit("gaus");
ERPt400_500->Fit("gaus");
ERPt500_750->Fit("gaus");

Int_t n = 11;
TF1* g[n];
Double_t mean[n], sigma[n], Mean[n], RMS[n], pt[n];

pt[0] = 30.;
pt[1] = 50.;
pt[2] = 70.;
pt[3] = 90.;
pt[4] = 125.;
pt[5] = 175.;
pt[6] = 225.;
pt[7] = 275.;
pt[8] = 350.;
pt[9] = 450.;
pt[10] = 600.;

TF1* g[0]  = ERPt20_40->GetFunction("gaus");
TF1* g[1]  = ERPt40_60->GetFunction("gaus");
TF1* g[2]  = ERPt60_80->GetFunction("gaus");
TF1* g[3]  = ERPt80_100->GetFunction("gaus");
TF1* g[4]  = ERPt100_150->GetFunction("gaus");
TF1* g[5]  = ERPt150_200->GetFunction("gaus");
TF1* g[6]  = ERPt200_250->GetFunction("gaus");
TF1* g[7]  = ERPt250_300->GetFunction("gaus");
TF1* g[8]  = ERPt300_400->GetFunction("gaus");
TF1* g[9]  = ERPt400_500->GetFunction("gaus");
TF1* g[10] = ERPt500_750->GetFunction("gaus");

Double_t Mean[0]  = ERPt20_40->GetMean();
Double_t Mean[1]  = ERPt40_60->GetMean();
Double_t Mean[2]  = ERPt60_80->GetMean();
Double_t Mean[3]  = ERPt80_100->GetMean();
Double_t Mean[4]  = ERPt100_150->GetMean();
Double_t Mean[5]  = ERPt150_200->GetMean();
Double_t Mean[6]  = ERPt200_250->GetMean();
Double_t Mean[7]  = ERPt250_300->GetMean();
Double_t Mean[8]  = ERPt300_400->GetMean();
Double_t Mean[9]  = ERPt400_500->GetMean();
Double_t Mean[10]  = ERPt500_750->GetMean();

Double_t RMS[0]  = ERPt20_40->GetRMS();
Double_t RMS[1]  = ERPt40_60->GetRMS();
Double_t RMS[2]  = ERPt60_80->GetRMS();
Double_t RMS[3]  = ERPt80_100->GetRMS();
Double_t RMS[4]  = ERPt100_150->GetRMS();
Double_t RMS[5]  = ERPt150_200->GetRMS();
Double_t RMS[6]  = ERPt200_250->GetRMS();
Double_t RMS[7]  = ERPt250_300->GetRMS();
Double_t RMS[8]  = ERPt300_400->GetRMS();
Double_t RMS[9]  = ERPt400_500->GetRMS();
Double_t RMS[10]  = ERPt500_750->GetRMS();

for (Int_t i=0; i<n; ++i) {
  mean[i] = g[i]->GetParameter(1);
  sigma[i] = g[i]->GetParameter(2);
}

TCanvas* c1 = new TCanvas;
c1->Divide(2,1);
c1->cd(1);
c1->cd(1)->SetGridx();
c1->cd(1)->SetGridy();
TGraph *gr1 = new TGraph ( n, pt, sigma );
TGraph *gr11 = new TGraph ( n, pt, RMS );
gr1->SetLineColor(kBlue);
gr1->SetLineWidth(2);
gr1->SetMinimum(0.02);
gr1->SetMaximum(0.16);
gr1->SetTitle("Resolution Delta(pT)/pT");
gr1->Draw("AC*");
gr11->SetMarkerStyle(21);
gr11->SetMarkerSize(0.5);
gr11->SetLineColor(kCyan);
gr11->Draw("CP");
// c1->Print("resolution.gif");
//TCanvas* c2 = new TCanvas;
//c2->SetGridx();
//c2->SetGridy();
c1->cd(2);
c1->cd(2)->SetGridx();
c1->cd(2)->SetGridy();
TGraph *gr2 = new TGraph ( n, pt, mean );
TGraph *gr12 = new TGraph ( n, pt, Mean );
gr2->SetMinimum(-0.25);
gr2->SetMaximum(0.01);
gr2->SetLineColor(kRed);
gr2->SetLineWidth(2);
gr2->SetTitle("Mean Delta(pT)/pT");
gr2->Draw("AC*");
gr12->SetMarkerStyle(21);
gr12->SetMarkerSize(0.5);
gr12->SetLineColor(kViolet);
gr12->Draw("CP");
//c2->Print("mean.gif");
c1->Print("resolutionEndcap.png");
}
