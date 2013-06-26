{
gStyle->SetOptStat(1111111);

TString nameOfHisto;
nameOfHisto= "CSC Muon Alignment Monitor";
TCanvas *mycanvas = new TCanvas("mycanvas",nameOfHisto);
_file0->cd("MuonAlignmentMonitor");

mycanvas->Divide(2,4);
mycanvas->Update();
mycanvas->cd(1);

nameOfHisto= "hprofLocalXCSC";
TH1F * h = (TH1F*) gROOT->FindObject(nameOfHisto);

h->Draw("");

mycanvas->cd(2);

nameOfHisto= "hprofGlobalRPhiCSC";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(3);

nameOfHisto= "hprofLocalYCSC";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(4);

nameOfHisto= "hprofGlobalRCSC";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(5);

nameOfHisto= "hprofLocalPhiCSC";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(6);

nameOfHisto= "hprofGlobalPhiCSC";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(7);

nameOfHisto= "hprofLocalThetaCSC";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(8);

nameOfHisto= "hprofGlobalThetaCSC";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

}
