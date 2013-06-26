#include <iostream.h>

int drawChamberDT(int wheel=0, int station = 1, int sector=1)
{
gStyle->SetOptStat(1111111);

char nameOfHisto[50];
sprintf(nameOfHisto, "W%dMB%1dS%1d",wheel,station,sector );
TCanvas *mycanvas = new TCanvas("mycanvas",nameOfHisto);
_file0->cd("MuonAlignmentMonitor");

mycanvas->Divide(2,4);
mycanvas->Update();
mycanvas->cd(1);

sprintf(nameOfHisto, "ResidualLocalX_W%dMB%1dS%1d",wheel,station,sector );
TH1F * h = (TH1F*) gROOT->FindObject(nameOfHisto);

h->Draw("");

mycanvas->cd(2);

sprintf(nameOfHisto, "ResidualGlobalRPhi_W%dMB%1dS%1d",wheel,station,sector );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(3);

sprintf(nameOfHisto, "ResidualLocalY_W%dMB%1dS%1d",wheel,station,sector );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(4);

sprintf(nameOfHisto, "ResidualGlobalZ_W%dMB%1dS%1d",wheel,station,sector );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(5);

sprintf(nameOfHisto, "ResidualLocalPhi_W%dMB%1dS%1d",wheel,station,sector );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(6);

sprintf(nameOfHisto, "ResidualGlobalPhi_W%dMB%1dS%1d",wheel,station,sector );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(7);

sprintf(nameOfHisto, "ResidualLocalTheta_W%dMB%1dS%1d",wheel,station,sector );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

mycanvas->cd(8);

sprintf(nameOfHisto, "ResidualGlobalTheta_W%dMB%1dS%1d",wheel,station,sector );
h = (TH1F*) gROOT->FindObject(nameOfHisto);
h->Draw("");

}
