#include <iostream.h>

int drawChamberCSC(int station = 1, int ring=1, int chamber=1)
{
gStyle->SetOptStat(1111111);

char nameOfHisto[50];
sprintf(nameOfHisto, "ME%dR%1dC%1d",station,ring,chamber );
TCanvas *mycanvas = new TCanvas("mycanvas",nameOfHisto);
_file0->cd("MuonAlignmentMonitor");

mycanvas->Divide(2,4);
mycanvas->Update();
mycanvas->cd(1);

sprintf(nameOfHisto, "ResidualLocalX_ME%dR%1dC%1d",station,ring,chamber );
TH1F * h = (TH1F*) gROOT->FindObject(nameOfHisto);

h->Draw("");

mycanvas->cd(2);

sprintf(nameOfHisto, "ResidualGlobalRPhi_ME%dR%1dC%1d",station,ring,chamber );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(3);

sprintf(nameOfHisto, "ResidualLocalY_ME%dR%1dC%1d",station,ring,chamber );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(4);

sprintf(nameOfHisto, "ResidualGlobalR_ME%dR%1dC%1d",station,ring,chamber );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(5);

sprintf(nameOfHisto, "ResidualLocalPhi_ME%dR%1dC%1d",station,ring,chamber );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(6);

sprintf(nameOfHisto, "ResidualGlobalPhi_ME%dR%1dC%1d",station,ring,chamber );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(7);

sprintf(nameOfHisto, "ResidualLocalTheta_ME%dR%1dC%1d",station,ring,chamber );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(8);

sprintf(nameOfHisto, "ResidualGlobalTheta_ME%dR%1dC%1d",station,ring,chamber );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

}
