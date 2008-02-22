{
gROOT->SetStyle("Plain");
gStyle->SetOptStat(000000);
string dir0 = "gsWithMaterial_AssociatorByHits";
string dir = "cutsGS_AssociatorByHits";
string dir2 = "cutsCKF_AssociatorByHits";
string plot = "effic";
double min = 0.70;
double max = 1.025;
TFile *_file0 = TFile::Open("valid_muon_1GeV.root");
TFile *_file1 = TFile::Open("valid_muon_10GeV.root");
TFile *_file2 = TFile::Open("valid_muon_100GeV.root");

//170pre12 fullsim
//TFile *_file3 = TFile::Open("/localscratch/azzi/FAMOS/FULLSIMVALID/CMSSW_1_7_0_pre12/src/valid_muon_10gev_RelVal_id.root");

TDirectory * dqm0 = _file0->Get("DQMData/Track");
TDirectory * CTF0 = dqm0->Get(dir.c_str());

TDirectory * dqm1 = _file1->Get("DQMData/Track");
TDirectory * CTF1 = dqm1->Get(dir.c_str());

TDirectory * dqm2 = _file2->Get("DQMData/Track");
TDirectory * CTF2 = dqm2->Get(dir.c_str());

//TDirectory * dqm3 = _file3->Get("DQMData/Track");
//TDirectory * CTF3 = dqm3->Get(dir2.c_str());

TCanvas * c = new TCanvas("c", "c") ;

TH1F * histo0 = CTF0->Get(plot.c_str());
TH1F * histo1 = CTF1->Get(plot.c_str());
TH1F * histo2 = CTF2->Get(plot.c_str());
//TH1F * histo3 = CTF3->Get(plot.c_str());


histo0->Scale(1);
histo1->Scale(1);
histo2->Scale(1);
//histo3->Scale(1);
histo0->GetYaxis().SetRangeUser(min,max);
histo0->GetYaxis().SetNdivisions(512,kTRUE);
histo0->Draw("P");
histo1->Draw("P,same");
histo2->Draw("P,same");
//histo3->Draw("P,same");

histo0->SetMarkerStyle(20);
histo0->SetMarkerColor(kBlack);

histo1->SetMarkerStyle(22);
histo1->SetMarkerColor(kBlue);

histo2->SetMarkerStyle(21);
histo2->SetMarkerColor(kRed);

//histo3->SetMarkerStyle(21);
//histo3->SetMarkerColor(kGreen);

c.Update();
c.SetGridx();
c.SetGridy();
TLegend * ll = new TLegend(0.5,0.1,0.8,0.25);
ll->AddEntry(histo0,"#mu pt=1 GeV","P");
ll->AddEntry(histo1,"#mu pt=10 GeV","P");
ll->AddEntry(histo2,"#mu pt=100 GeV","P");
//ll->AddEntry(histo3,"#mu pt=10 GeV","P");

ll->SetFillColor(kWhite);
ll->Draw();
c.SaveAs( "muon_eff.gif" );
//c.Clear();

}
