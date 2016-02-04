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
TH2F * h2 = (TH2F*) gROOT->FindObject(nameOfHisto);

h2->Draw("COLZ");

mycanvas->cd(2);

nameOfHisto= "hprofGlobalPositionCSC";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(3);

nameOfHisto= "hprofLocalPositionRmsCSC";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(4);

nameOfHisto= "hprofGlobalPositionRmsCSC";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(5);

nameOfHisto= "hprofLocalAngleCSC";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(6);

nameOfHisto= "hprofGlobalAngleCSC";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(7);

nameOfHisto= "hprofLocalAngleRmsCSC";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(8);

nameOfHisto= "hprofGlobalAngleRmsCSC";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

}
