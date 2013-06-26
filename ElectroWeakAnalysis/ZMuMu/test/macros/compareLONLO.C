{

gROOT->Reset();
gROOT->SetStyle("Plain");

TCanvas *c1 = new TCanvas ("c1" , "Pythia6 vs MCatNLO");

int  sec_LO = 19440;
int  sec_NLO = 22740;
double w_LO_over_NLO= double (19440) / double (22740)  ;


TFile *NLO = TFile::Open("ZNLO_10pb.root");


//mass

TH1D * ZMCNLOmass = (TH1D*) NLO->Get("zHistos/ZMCHisto/ZMCMass");
ZMCNLOmass->SetLineColor(kBlue);
ZMCNLOmass->SetStats(kFALSE);
ZMCNLOmass-> Draw();

TFile *LO = TFile::Open("ZLO_10pb.root");

TH1D * ZMCLOmass = (TH1D*) LO->Get("zHistos/ZMCHisto/ZMCMass");
ZMCLOmass->SetLineColor(kRed);

ZMCLOmass->Draw("SAME");
leg = new TLegend(.6,.6,0.9,0.9);
leg->AddEntry(ZMCNLOmass,"MCatNLO","l");
leg->AddEntry(ZMCLOmass,"Pythia6","l");
leg->SetFillColor(0);
leg->SetBorderSize(0);

leg->Draw("SAME");

c1.SaveAs("LO_versus_NLO/ZMass_LO_vs_NLO_at10pb.eps");


ZMCNLOmass->Scale(w_LO_over_NLO);
ZMCNLOmass->Draw("");
ZMCLOmass->Draw("SAME");
leg->Draw("SAME");
c1.SaveAs("LO_versus_NLO/ZPt_LO_vs_NLO_at10pb_normalized.eps");


  //pt



c1->SetLogy(1);

TH1D * ZMCNLOpt = (TH1D*) NLO->Get("zHistos/ZMCHisto/ZMCPt");
ZMCNLOpt->SetLineColor(kBlue);
ZMCNLOpt->Rebin(2);



TFile *LO = TFile::Open("ZLO_10pb.root");

TH1D * ZMCLOpt = (TH1D*) LO->Get("zHistos/ZMCHisto/ZMCPt");
ZMCLOpt->SetLineColor(kRed);
ZMCLOpt->Rebin(2);

ZMCLOpt->Draw();
ZMCLOpt->SetStats(kFALSE);
ZMCNLOpt-> Draw("same");

leg = new TLegend(.6,.6,0.9,0.9);
leg->AddEntry(ZMCNLOpt,"MCatNLO","l");
leg->AddEntry(ZMCLOpt,"Pythia6","l");
leg->SetFillColor(0);
leg->SetBorderSize(0);

leg->Draw("SAME");

c1->SaveAs("LO_versus_NLO/ZPt_LO_vs_NLO_at10pb.eps");

// L0/NLO ratio
TH1D numLO = TH1D( *ZMCLOpt);
numLO.Sumw2();

TH1D denNLO = TH1D( *ZMCNLOpt);
denNLO.Sumw2();

numLO.Divide(&denNLO);
numLO.Draw("b");
denNLO.Divide(&denNLO);
denNLO.Draw("same");
c1->SetLogy(0);
c1.SaveAs("LO_versus_NLO/ratioLO_versus_NLO_pt_at10pb.eps");

numLO->Scale(1. / w_LO_over_NLO);
c1.SaveAs("LO_versus_NLO/ratioLO_versus_NLO_Pt_LO_vs_NLO_at10pb_normalized.eps");


ZMCNLOpt->Scale(w_LO_over_NLO);
ZMCLOpt->Draw("");
ZMCNLOpt->Draw("same");
leg->Draw("SAME");
c1->SetLogy(1);
c1.SaveAs("LO_versus_NLO_Pt_LO_vs_NLO_at10pb_normalized.eps");



  //rapidity

c1->SetLogy(0);

TH1D * ZMCNLOrapidity = (TH1D*) NLO->Get("zHistos/ZMCHisto/ZMCRapidity");
ZMCNLOrapidity->SetLineColor(kBlue);
ZMCNLOrapidity->Rebin(2);
ZMCNLOrapidity-> Draw();
ZMCNLOrapidity->SetStats(kFALSE);


TFile *LO = TFile::Open("ZLO_10pb.root");

TH1D * ZMCLOrapidity = (TH1D*) LO->Get("zHistos/ZMCHisto/ZMCRapidity");
ZMCLOrapidity->SetLineColor(kRed);
ZMCLOrapidity->Rebin(2);
ZMCLOrapidity->Draw();
ZMCLOrapidity->SetStats(kFALSE);
ZMCNLOrapidity-> Draw("SAME");
leg = new TLegend(.7,.7,0.9,0.9);
leg->AddEntry(ZMCNLOrapidity,"MCatNLO","l");
leg->AddEntry(ZMCLOrapidity,"Pythia6","l");
leg->SetFillColor(0);
leg->SetBorderSize(0);

leg->Draw("SAME");

c1.SaveAs("LO_versus_NLO/ZRapidity_LO_vs_NLO_at10pb.eps");

// L0/NLO ratio
TH1D numLO = TH1D( *ZMCLOrapidity);
numLO.Sumw2();

TH1D denNLO = TH1D( *ZMCNLOrapidity);
denNLO.Sumw2();

numLO.Divide(&denNLO);
numLO.Draw("b");
denNLO.Divide(&denNLO);
denNLO.Draw("same");
c1->SetLogy(0);
c1.SaveAs("LO_versus_NLO/ratioLO_versus_NLO_rapidity_at10pb.eps");

numLO->Scale(1. / w_LO_over_NLO);
c1.SaveAs("LO_versus_NLO/ratioLO_versus_NLO_rapidity_LO_vs_NLO_at10pb_normalized.eps");





ZMCLOrapidity->Draw();
ZMCNLOrapidity->Scale(w_LO_over_NLO);
ZMCNLOrapidity->Draw("SAME");
leg->Draw("SAME");
c1->SetLogy(1);
c1.SaveAs("LO_versus_NLO/ZRapidity_LO_vs_NLO_at10pb_normalized.eps");




TFile * ZToLL_file = new TFile("CompareLONLO.root","recreate");




ZToLL_file->Close();

}
