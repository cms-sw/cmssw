#include <vector>

void PlotGraphs( TH1F* fast, TH1F* full) {

  fast->SetMarkerColor(4);						
  fast->SetLineColor(4);						  
  fast->SetLineWidth(2);						  
  fast->SetLineStyle(2);
  fast->Draw();


  full->SetMarkerStyle(25);						
  full->SetMarkerColor(2);						
  full->SetLineColor(2);						  
  full->SetLineWidth(2);						  
  full->Draw("same");
  
}

void Efficiency() {
  
  gROOT->Reset();
  TFile *f = new TFile("testTrackingIterations.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  std::vector<const char *> titleFull;
  std::vector<const char *> titleFast;
  std::vector<const char *> titleGen;
  titleFull.push_back("eff1Full_p_0_1");
  titleFull.push_back("eff1Full_p_1_2");
  titleFull.push_back("eff1Full_p_2_3");
  titleFull.push_back("eff1Full_p_3_4");
  titleFull.push_back("eff1Full_p_4_5");
  titleFull.push_back("eff1Full_p_5_6");
  titleFull.push_back("eff1Full_p_6_7");
  titleFull.push_back("eff1Full_p_7_8");
  titleFull.push_back("eff1Full_p_8_9");
  titleFull.push_back("eff1Full_p_9_10");
  titleFast.push_back("eff1Fast_p_0_1");
  titleFast.push_back("eff1Fast_p_1_2");
  titleFast.push_back("eff1Fast_p_2_3");
  titleFast.push_back("eff1Fast_p_3_4");
  titleFast.push_back("eff1Fast_p_4_5");
  titleFast.push_back("eff1Fast_p_5_6");
  titleFast.push_back("eff1Fast_p_6_7");
  titleFast.push_back("eff1Fast_p_7_8");
  titleFast.push_back("eff1Fast_p_8_9");
  titleFast.push_back("eff1Fast_p_9_10");
  titleGen.push_back("eff1Gen_p_0_1");
  titleGen.push_back("eff1Gen_p_1_2");
  titleGen.push_back("eff1Gen_p_2_3");
  titleGen.push_back("eff1Gen_p_3_4");
  titleGen.push_back("eff1Gen_p_4_5");
  titleGen.push_back("eff1Gen_p_5_6");
  titleGen.push_back("eff1Gen_p_6_7");
  titleGen.push_back("eff1Gen_p_7_8");
  titleGen.push_back("eff1Gen_p_8_9");
  titleGen.push_back("eff1Gen_p_9_10");

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(3,4);
  for (unsigned imom=1;imom<11;++imom) {  
    genEtaP->ProjectionX(titleGen[imom-1],10*(imom-1)+1,10*imom+1);
    eff3Full->ProjectionX(titleFull[imom-1],10*(imom-1)+1,10*imom+1);
    eff3Fast->ProjectionX(titleFast[imom-1],10*(imom-1)+1,10*imom+1);
    TH1F* fast = (TH1F*) gDirectory->Get(titleFast[imom-1]);
    TH1F* full = (TH1F*) gDirectory->Get(titleFull[imom-1]);
    TH1F* gen = (TH1F*) gDirectory->Get(titleGen[imom-1]);
    fast->Divide(gen);
    full->Divide(gen);
    c->cd(imom);
    PlotGraphs(fast,full);
  }  
  
  genEtaP->ProjectionX();
  eff3Full->ProjectionX();
  eff3Fast->ProjectionX();
  TH1F* fast = (TH1F*) gDirectory->Get("eff3Fast_px");
  TH1F* full = (TH1F*) gDirectory->Get("eff3Full_px");
  TH1F* gen = (TH1F*) gDirectory->Get("genEtaP_px");
  fast->Divide(gen);
  full->Divide(gen);
  c->cd(11);
  PlotGraphs(fast,full);
}
