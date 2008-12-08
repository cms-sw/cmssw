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

void Efficiency(unsigned int iter) {
  
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
  TH2F* effFull;
  TH2F* effFast;
  TH2F* genPlot;
  genPlot = (TH2F*) gDirectory->Get("genEtaP");
  if ( iter == 1 ) { 
    effFull = (TH2F*) gDirectory->Get("eff1Full");
    effFast = (TH2F*) gDirectory->Get("eff1Fast");
  } else if ( iter == 2 ) { 
    effFull = (TH2F*) gDirectory->Get("eff2Full");
    effFast = (TH2F*) gDirectory->Get("eff2Fast");
  } else if ( iter == 3 ) { 
    effFull = (TH2F*) gDirectory->Get("eff3Full");
    effFast = (TH2F*) gDirectory->Get("eff3Fast");
  } else if ( iter == 11 ) { 
    effFull = (TH2F*) gDirectory->Get("eff1Full1");
    effFast = (TH2F*) gDirectory->Get("eff1Fast1");
  } else if ( iter == 12 ) { 
    effFull = (TH2F*) gDirectory->Get("eff1Full2");
    effFast = (TH2F*) gDirectory->Get("eff1Fast2");
  }
  for (unsigned imom=1;imom<11;++imom) {  
    genPlot->ProjectionX(titleGen[imom-1],10*(imom-1)+1,10*imom+1);
    effFull->ProjectionX(titleFull[imom-1],10*(imom-1)+1,10*imom+1);
    effFast->ProjectionX(titleFast[imom-1],10*(imom-1)+1,10*imom+1);
    TH1F* fast = (TH1F*) gDirectory->Get(titleFast[imom-1]);
    TH1F* full = (TH1F*) gDirectory->Get(titleFull[imom-1]);
    TH1F* gen = (TH1F*) gDirectory->Get(titleGen[imom-1]);
    fast->Divide(gen);
    full->Divide(gen);
    c->cd(imom);
    PlotGraphs(fast,full);
  }  
  
  TH1F* fast;
  TH1F* full;
  TH1F* gen;
  TH1F* fastp;
  TH1F* fullp;
  TH1F* genp;
  genPlot->ProjectionX();
  effFull->ProjectionX();
  effFast->ProjectionX();
  genPlot->ProjectionY();
  effFull->ProjectionY();
  effFast->ProjectionY();
  if ( iter == 1 ) { 
    fast = (TH1F*) gDirectory->Get("eff1Fast_px");
    full = (TH1F*) gDirectory->Get("eff1Full_px");
    fastp = (TH1F*) gDirectory->Get("eff1Fast_py");
    fullp = (TH1F*) gDirectory->Get("eff1Full_py");
  } else if ( iter == 2 ) { 
    fast = (TH1F*) gDirectory->Get("eff2Fast_px");
    full = (TH1F*) gDirectory->Get("eff2Full_px");
    fastp = (TH1F*) gDirectory->Get("eff2Fast_py");
    fullp = (TH1F*) gDirectory->Get("eff2Full_py");
  } else if ( iter == 3 ) { 
    fast = (TH1F*) gDirectory->Get("eff3Fast_px");
    full = (TH1F*) gDirectory->Get("eff3Full_px");
    fastp = (TH1F*) gDirectory->Get("eff3Fast_py");
    fullp = (TH1F*) gDirectory->Get("eff3Full_py");
  } else if ( iter == 11 ) { 
    fast = (TH1F*) gDirectory->Get("eff1Fast1_px");
    full = (TH1F*) gDirectory->Get("eff1Full1_px");
    fastp = (TH1F*) gDirectory->Get("eff1Fast1_py");
    fullp = (TH1F*) gDirectory->Get("eff1Full1_py");
  } else if ( iter == 12 ) { 
    fast = (TH1F*) gDirectory->Get("eff1Fast2_px");
    full = (TH1F*) gDirectory->Get("eff1Full2_px");
    fastp = (TH1F*) gDirectory->Get("eff1Fast2_py");
    fullp = (TH1F*) gDirectory->Get("eff1Full2_py");
  }
  gen = (TH1F*) gDirectory->Get("genEtaP_px");
  fast->Divide(gen);
  full->Divide(gen);
  c->cd(11);
  PlotGraphs(fast,full);

  genp = (TH1F*) gDirectory->Get("genEtaP_py");
  fastp->Divide(genp);
  fullp->Divide(genp);
  c->cd(12);
  PlotGraphs(fastp,fullp);

}

void totalEfficiency(unsigned int iter) {
  
  gROOT->Reset();
  TFile *f = new TFile("testTrackingIterations.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(1,2);
  TH2F* iter1Fast;
  TH2F* iter2Fast;
  TH2F* iter3Fast;
  TH2F* iter1Full;
  TH2F* iter2Full;
  TH2F* iter3Full;
  TH2F* genPlot;
  genPlot = (TH2F*) gDirectory->Get("genEtaP");
  iter1Full = (TH2F*) gDirectory->Get("eff1Full");
  iter1Fast = (TH2F*) gDirectory->Get("eff1Fast");
  iter2Full = (TH2F*) gDirectory->Get("eff2Full");
  iter2Fast = (TH2F*) gDirectory->Get("eff2Fast");
  iter3Full = (TH2F*) gDirectory->Get("eff3Full");
  iter3Fast = (TH2F*) gDirectory->Get("eff3Fast");
  
  TH1F* fast1;
  TH1F* full1;
  TH1F* fast2;
  TH1F* full2;
  TH1F* fast3;
  TH1F* full3;
  TH1F* gen;
  TH1F* fastp1;
  TH1F* fullp1;
  TH1F* fastp2;
  TH1F* fullp2;
  TH1F* fastp3;
  TH1F* fullp3;
  TH1F* genp;
  genPlot->ProjectionX();
  iter1Full->ProjectionX("iter1Full_px");
  iter1Fast->ProjectionX("iter1Fast_px");
  iter2Full->ProjectionX("iter2Full_px");
  iter2Fast->ProjectionX("iter2Fast_px");
  iter3Full->ProjectionX("iter3Full_px");
  iter3Fast->ProjectionX("iter3Fast_px");
  genPlot->ProjectionY();
  iter1Full->ProjectionY("iter1Full_py");
  iter1Fast->ProjectionY("iter1Fast_py");
  iter2Full->ProjectionY("iter2Full_py");
  iter2Fast->ProjectionY("iter2Fast_py");
  iter3Full->ProjectionY("iter3Full_py");
  iter3Fast->ProjectionY("iter3Fast_py");
  fast1  = (TH1F*) gDirectory->Get("iter1Fast_px");
  full1  = (TH1F*) gDirectory->Get("iter1Full_px");
  fastp1 = (TH1F*) gDirectory->Get("iter1Fast_py");
  fullp1 = (TH1F*) gDirectory->Get("iter1Full_py");
  fast2  = (TH1F*) gDirectory->Get("iter2Fast_px");
  full2  = (TH1F*) gDirectory->Get("iter2Full_px");
  fastp2 = (TH1F*) gDirectory->Get("iter2Fast_py");
  fullp2 = (TH1F*) gDirectory->Get("iter2Full_py");
  fast3  = (TH1F*) gDirectory->Get("iter3Fast_px");
  full3  = (TH1F*) gDirectory->Get("iter3Full_px");
  fastp3 = (TH1F*) gDirectory->Get("iter3Fast_py");
  fullp3 = (TH1F*) gDirectory->Get("iter3Full_py");
  gen = (TH1F*) gDirectory->Get("genEtaP_px");
  genp = (TH1F*) gDirectory->Get("genEtaP_py");

  if ( iter == 2 ) fast1 = fast2;
  if ( iter == 3 ) fast1 = fast3;
  if ( iter > 11 ) fast1->Add(fast2);
  if ( iter > 12 ) fast1->Add(fast3);
  fast1->Divide(gen);
  if ( iter == 2 ) full1 = full2;
  if ( iter == 3 ) full1 = full3;
  if ( iter > 11 ) full1->Add(full2);
  if ( iter > 12 ) full1->Add(full3);
  full1->Divide(gen);
  c->cd(1);
  PlotGraphs(fast1,full1);

  if ( iter == 2 ) fastp1 = fastp2;
  if ( iter == 3 ) fastp1 = fastp3;
  if ( iter > 11 ) fastp1->Add(fastp2);
  if ( iter > 12 ) fastp1->Add(fastp3);
  fastp1->Divide(genp);
  if ( iter == 2 ) fullp1 = fullp2;
  if ( iter == 3 ) fullp1 = fullp3;
  if ( iter > 11 ) fullp1->Add(fullp2);
  if ( iter > 12 ) fullp1->Add(fullp3);
  fullp1->Divide(genp);
  c->cd(2);
  PlotGraphs(fastp1,fullp1);

}
