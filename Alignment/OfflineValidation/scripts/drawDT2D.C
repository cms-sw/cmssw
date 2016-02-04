{
gStyle->SetOptStat(00000000);

TString nameOfHisto;
nameOfHisto= "DT Muon Alignment Monitor";
TCanvas *mycanvas = new TCanvas("mycanvas",nameOfHisto);
_file0->cd("MuonAlignmentMonitor");

mycanvas->Divide(2,4);
mycanvas->Update();
mycanvas->cd(1);

nameOfHisto= "hprofLocalPositionDT";
TH2F * h2 = (TH2F*) gROOT->FindObject(nameOfHisto);

h2->Draw("COLZ");

mycanvas->cd(2);

nameOfHisto= "hprofGlobalPositionDT";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(3);

nameOfHisto= "hprofLocalPositionRmsDT";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(4);

nameOfHisto= "hprofGlobalPositionRmsDT";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(5);

nameOfHisto= "hprofLocalAngleDT";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(6);

nameOfHisto= "hprofGlobalAngleDT";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(7);

nameOfHisto= "hprofLocalAngleRmsDT";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

mycanvas->cd(8);

nameOfHisto= "hprofGlobalAngleRmsDT";
h2 = (TH2F*) gROOT->FindObject(nameOfHisto);
h2->Draw("COLZ");

}
