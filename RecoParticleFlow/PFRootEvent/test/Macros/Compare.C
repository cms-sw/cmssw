#include <vector>

void PlotHisto(char* hname, TFile* f1, TFile* f2)
{
TH1F* h1 = (TH1F*) f1->Get(hname);
TH1F* h2 = (TH1F*) f2->Get(hname);
h1->SetTitle(hname);
h1->SetStats(0);
h1->SetMinimum(0);

h1->Scale( h2->GetEntries()/h1->GetEntries() );

h1->SetMarkerStyle(22);
h1->SetMarkerSize(0.6);
h1->SetMarkerColor(2);
h1->Draw("error");
h2->SetLineColor(4);
h2->SetLineWidth(1.5);
h2->Draw("same");
}

void Compare()
{
  gROOT->Reset();
  TFile *f1 = new TFile("pfjetBenchmarkFast.root");
  TFile *f2 = new TFile("pfjetBenchmarkFull.root");
  vector< string > hists;
  hists.push_back( "BRPt20_40") ;
  hists.push_back( "BRPt40_60");
  hists.push_back( "BRPt60_80");
  hists.push_back( "BRPt80_100");
  hists.push_back( "BRPt100_150");
  hists.push_back( "BRPt150_200");
  hists.push_back( "BRPt200_250");
  hists.push_back( "BRPt250_300");
  hists.push_back( "BRPt300_400");
  hists.push_back( "BRPt400_500");
  hists.push_back( "BRPt500_750");

  /*
  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(3,4);
  for( unsigned i=0; i<hists.size(); ++i) {
    c->cd(i+1);
    PlotHisto( hists[i].c_str(), f1, f2 );
  }
  */

  TCanvas *c1 = new TCanvas("c1","",1000,600);
  c1->Divide(2,2);
  string hist1 = "BRneut";
  c1->cd(1);
  PlotHisto( hist1.c_str(), f1, f2 );
  string hist2 = "BRCHE";
  c1->cd(2);
  PlotHisto( hist2.c_str(), f1, f2 );
  string hist3 = "ERneut";
  c1->cd(3);
  PlotHisto( hist3.c_str(), f1, f2 );
  string hist4 = "ERCHE";
  c1->cd(4);
  PlotHisto( hist4.c_str(), f1, f2 );
}
