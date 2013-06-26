{
gStyle->SetOptStat(1111111);

TString nameOfHisto;
nameOfHisto= "DT Muon Alignment Monitor";
TCanvas *mycanvas = new TCanvas("mycanvas",nameOfHisto);
_file0->cd("MuonAlignmentMonitor");

mycanvas->Divide(2,4);
mycanvas->Update();
mycanvas->cd(1);

nameOfHisto= "hprofLocalXDT";
TH1F * h = (TH1F*) gROOT->FindObject(nameOfHisto);

h->Draw("");

mycanvas->cd(2);

nameOfHisto= "hprofGlobalRPhiDT";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(3);

nameOfHisto= "hprofLocalYDT";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(4);

nameOfHisto= "hprofGlobalZDT";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(5);

nameOfHisto= "hprofLocalPhiDT";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(6);

nameOfHisto= "hprofGlobalPhiDT";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(7);

nameOfHisto= "hprofLocalThetaDT";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(8);

nameOfHisto= "hprofGlobalThetaDT";
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

}
