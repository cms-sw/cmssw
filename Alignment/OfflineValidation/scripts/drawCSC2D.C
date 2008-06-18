{
gStyle->SetOptStat(00000000);

TString nameOfHisto;
nameOfHisto= "CSC Muon Alignment Monitor";
TCanvas *mycanvas = new TCanvas("mycanvas",nameOfHisto);
_file0->cd("MuonAlignmentMonitor");

mycanvas->Divide(2,4);
mycanvas->Update();
mycanvas->cd(1);

nameOfHisto= "hprofLocalPositionCSC";
TH2F * h = (TH2F*) gROOT->FindObject(nameOfHisto);

h->Draw("COLZ");

mycanvas->cd(2);

nameOfHisto= "hprofGlobalPositionCSC";
h = (TH2F*) gROOT->FindObject(nameOfHisto);
h->Draw("COLZ");

mycanvas->cd(3);

nameOfHisto= "hprofLocalPositionRmsCSC";
h = (TH2F*) gROOT->FindObject(nameOfHisto);
h->Draw("COLZ");

mycanvas->cd(4);

nameOfHisto= "hprofGlobalPositionRmsCSC";
h = (TH2F*) gROOT->FindObject(nameOfHisto);
h->Draw("COLZ");

mycanvas->cd(5);

nameOfHisto= "hprofLocalAngleCSC";
h = (TH2F*) gROOT->FindObject(nameOfHisto);
h->Draw("COLZ");

mycanvas->cd(6);

nameOfHisto= "hprofGlobalAngleCSC";
h = (TH2F*) gROOT->FindObject(nameOfHisto);
h->Draw("COLZ");

mycanvas->cd(7);

nameOfHisto= "hprofLocalAngleRmsCSC";
h = (TH2F*) gROOT->FindObject(nameOfHisto);
h->Draw("COLZ");

mycanvas->cd(8);

nameOfHisto= "hprofGlobalAngleRmsCSC";
h = (TH2F*) gROOT->FindObject(nameOfHisto);
h->Draw("COLZ");

}
