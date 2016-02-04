#include <vector>

void PlotGraphs( TH1F* full, TH1F* fast) {

  full->SetMarkerColor(4);						
  full->SetLineColor(4);//blue						  
  full->SetLineWidth(2);						  
  full->SetLineStyle(3);
  //  full->SetMaximum(0.11);
  full->Draw();


  fast->SetMarkerStyle(25);						
  fast->SetMarkerColor(2);						
  fast->SetLineColor(2);//red						  
  fast->SetLineWidth(2);						  
  fast->Draw("same");
  
}

void Efficiency(unsigned int iter) {
  
  gROOT->Reset();
  TFile *f = new TFile("testTrackingIterations.root");
  //  TFile *f = new TFile("testTrackingIterations_fullstat.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  std::vector<char *> titleFull;
  std::vector<char *> titleFast;
  std::vector<char *> titleGen;
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
  if ( iter == 0 ) { 
    effFull = (TH2F*) gDirectory->Get("eff0Full");
    effFast = (TH2F*) gDirectory->Get("eff0Fast");
  } else if ( iter == 1 ) { 
    effFull = (TH2F*) gDirectory->Get("eff1Full");
    effFast = (TH2F*) gDirectory->Get("eff1Fast");
  } else if ( iter == 2 ) { 
    effFull = (TH2F*) gDirectory->Get("eff2Full");
    effFast = (TH2F*) gDirectory->Get("eff2Fast");
  } else if ( iter == 3 ) { 
    effFull = (TH2F*) gDirectory->Get("eff3Full");
    effFast = (TH2F*) gDirectory->Get("eff3Fast");
  } else if ( iter == 4 ) { 
    effFull = (TH2F*) gDirectory->Get("eff4Full");
    effFast = (TH2F*) gDirectory->Get("eff4Fast");
  } else if ( iter == 5 ) { 
    effFull = (TH2F*) gDirectory->Get("eff5Full");
    effFast = (TH2F*) gDirectory->Get("eff5Fast");
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
    PlotGraphs(full,fast);
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
  if ( iter == 0 ) { 
    fast = (TH1F*) gDirectory->Get("eff0Fast_px");
    full = (TH1F*) gDirectory->Get("eff0Full_px");
    fastp = (TH1F*) gDirectory->Get("eff0Fast_py");
    fullp = (TH1F*) gDirectory->Get("eff0Full_py");
  }else if ( iter == 1 ) { 
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
  } else if ( iter == 4 ) { 
    fast = (TH1F*) gDirectory->Get("eff4Fast_px");
    full = (TH1F*) gDirectory->Get("eff4Full_px");
    fastp = (TH1F*) gDirectory->Get("eff4Fast_py");
    fullp = (TH1F*) gDirectory->Get("eff4Full_py");
  } else if ( iter == 5 ) { 
    fast = (TH1F*) gDirectory->Get("eff5Fast_px");
    full = (TH1F*) gDirectory->Get("eff5Full_px");
    fastp = (TH1F*) gDirectory->Get("eff5Fast_py");
    fullp = (TH1F*) gDirectory->Get("eff5Full_py");
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
  PlotGraphs(full,fast);

  genp = (TH1F*) gDirectory->Get("genEtaP_py");
  fastp->Divide(genp);
  fullp->Divide(genp);
  c->cd(12);
  PlotGraphs(fullp,fastp);

}

void Hits(unsigned int iter) {
  
  gROOT->Reset();
    TFile *f = new TFile("testTrackingIterations.root");
    //TFile *f = new TFile("testTrackingIterations_fullstat.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  std::vector<char *> titleFull;
  std::vector<char *> titleFast;
  titleFull.push_back("HitsFull_p_0_2");
  titleFull.push_back("HitsFull_p_2_4");
  titleFull.push_back("HitsFull_p_4_6");
  titleFull.push_back("HitsFull_p_6_8");
  titleFull.push_back("HitsFull_p_8_10");
  titleFull.push_back("HitsFull_Eta_0.0_0.5");
  titleFull.push_back("HitsFull_Eta_0.5_1.0");
  titleFull.push_back("HitsFull_Eta_1.0_1.5");
  titleFull.push_back("HitsFull_Eta_1.5_2.0");
  titleFull.push_back("HitsFull_Eta_2.0_2.8");
  titleFast.push_back("HitsFast_p_0_2");
  titleFast.push_back("HitsFast_p_2_4");
  titleFast.push_back("HitsFast_p_4_6");
  titleFast.push_back("HitsFast_p_6_8");
  titleFast.push_back("HitsFast_p_8_10");
  titleFast.push_back("HitsFast_Eta_0.0_0.6");
  titleFast.push_back("HitsFast_Eta_0.6_1.2");
  titleFast.push_back("HitsFast_Eta_1.2_1.8");
  titleFast.push_back("HitsFast_Eta_1.8_2.4");
  titleFast.push_back("HitsFast_Eta_2.4_3.0");

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(3,4);
  TH2F* hitsFullP;
  TH2F* hitsFastP;
  TH2F* hitsFullEta;
  TH2F* hitsFastEta;
  if ( iter == 0 ) { 
    hitsFullP = (TH2F*) gDirectory->Get("Hits0PFull");
    hitsFastP = (TH2F*) gDirectory->Get("Hits0PFast");
    hitsFullEta = (TH2F*) gDirectory->Get("Hits0EtaFull");
    hitsFastEta = (TH2F*) gDirectory->Get("Hits0EtaFast");
  }else if ( iter == 1 ) { 
    hitsFullP = (TH2F*) gDirectory->Get("Hits1PFull");
    hitsFastP = (TH2F*) gDirectory->Get("Hits1PFast");
    hitsFullEta = (TH2F*) gDirectory->Get("Hits1EtaFull");
    hitsFastEta = (TH2F*) gDirectory->Get("Hits1EtaFast");
  } else if ( iter == 2 ) { 
    hitsFullP = (TH2F*) gDirectory->Get("Hits2PFull");
    hitsFastP = (TH2F*) gDirectory->Get("Hits2PFast");
    hitsFullEta = (TH2F*) gDirectory->Get("Hits2EtaFull");
    hitsFastEta = (TH2F*) gDirectory->Get("Hits2EtaFast");
  } else if ( iter == 3 ) { 
    hitsFullP = (TH2F*) gDirectory->Get("Hits3PFull");
    hitsFastP = (TH2F*) gDirectory->Get("Hits3PFast");
    hitsFullEta = (TH2F*) gDirectory->Get("Hits3EtaFull");
    hitsFastEta = (TH2F*) gDirectory->Get("Hits3EtaFast");
  } else if ( iter == 4 ) { 
    hitsFullP = (TH2F*) gDirectory->Get("Hits4PFull");
    hitsFastP = (TH2F*) gDirectory->Get("Hits4PFast");
    hitsFullEta = (TH2F*) gDirectory->Get("Hits4EtaFull");
    hitsFastEta = (TH2F*) gDirectory->Get("Hits4EtaFast");
  } else if ( iter == 5 ) { 
    hitsFullP = (TH2F*) gDirectory->Get("Hits5PFull");
    hitsFastP = (TH2F*) gDirectory->Get("Hits5PFast");
    hitsFullEta = (TH2F*) gDirectory->Get("Hits5EtaFull");
    hitsFastEta = (TH2F*) gDirectory->Get("Hits5EtaFast");
  }
  for (unsigned imom=1;imom<6;++imom) {  
    hitsFullP->ProjectionY(titleFull[imom-1],20*(imom-1)+1,20*imom+1);
    hitsFastP->ProjectionY(titleFast[imom-1],20*(imom-1)+1,20*imom+1);
    hitsFullEta->ProjectionY(titleFull[10-imom],3*(imom-1),max((unsigned int)28,30-3*(imom-1)));
    hitsFastEta->ProjectionY(titleFast[10-imom],3*(imom-1),max((unsigned int)28,30-3*(imom-1)));
    hitsFullEta->ProjectionY("cacaFull",3*imom,30-3*imom);
    hitsFastEta->ProjectionY("cacaFast",3*imom,30-3*imom);

    TH1F* fastP = (TH1F*) gDirectory->Get(titleFast[imom-1]);
    TH1F* fullP = (TH1F*) gDirectory->Get(titleFull[imom-1]);
    TH1F* fastEta = (TH1F*) gDirectory->Get(titleFast[10-imom]);
    TH1F* fullEta = (TH1F*) gDirectory->Get(titleFull[10-imom]);
    TH1F* fastEta2 = (TH1F*) gDirectory->Get("cacaFast");
    TH1F* fullEta2 = (TH1F*) gDirectory->Get("cacaFull");
    fastEta->Add(fastEta2,-1);
    fullEta->Add(fullEta2,-1);
    
    c->cd(imom);
    PlotGraphs(fullP,fastP);
    c->cd(12-imom);
    PlotGraphs(fullEta,fastEta);
  }  
  
}

void Layers(unsigned int iter) {
  
  gROOT->Reset();
    TFile *f = new TFile("testTrackingIterations.root");
  //TFile *f = new TFile("testTrackingIterations_fullstat.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  std::vector<char *> titleFull;
  std::vector<char *> titleFast;
  titleFull.push_back("LayersFull_p_0_2");
  titleFull.push_back("LayersFull_p_2_4");
  titleFull.push_back("LayersFull_p_4_6");
  titleFull.push_back("LayersFull_p_6_8");
  titleFull.push_back("LayersFull_p_8_10");
  titleFull.push_back("LayersFull_Eta_0.0_0.5");
  titleFull.push_back("LayersFull_Eta_0.5_1.0");
  titleFull.push_back("LayersFull_Eta_1.0_1.5");
  titleFull.push_back("LayersFull_Eta_1.5_2.0");
  titleFull.push_back("LayersFull_Eta_2.0_2.8");
  titleFast.push_back("LayersFast_p_0_2");
  titleFast.push_back("LayersFast_p_2_4");
  titleFast.push_back("LayersFast_p_4_6");
  titleFast.push_back("LayersFast_p_6_8");
  titleFast.push_back("LayersFast_p_8_10");
  titleFast.push_back("LayersFast_Eta_0.0_0.6");
  titleFast.push_back("LayersFast_Eta_0.6_1.2");
  titleFast.push_back("LayersFast_Eta_1.2_1.8");
  titleFast.push_back("LayersFast_Eta_1.8_2.4");
  titleFast.push_back("LayersFast_Eta_2.4_3.0");

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(3,4);
  TH2F* layersFullP;
  TH2F* layersFastP;
  TH2F* layersFullEta;
  TH2F* layersFastEta;
  if ( iter == 0 ) { 
    layersFullP = (TH2F*) gDirectory->Get("Layers0PFull");
    layersFastP = (TH2F*) gDirectory->Get("Layers0PFast");
    layersFullEta = (TH2F*) gDirectory->Get("Layers0EtaFull");
    layersFastEta = (TH2F*) gDirectory->Get("Layers0EtaFast");
  } else if ( iter == 1 ) { 
    layersFullP = (TH2F*) gDirectory->Get("Layers1PFull");
    layersFastP = (TH2F*) gDirectory->Get("Layers1PFast");
    layersFullEta = (TH2F*) gDirectory->Get("Layers1EtaFull");
    layersFastEta = (TH2F*) gDirectory->Get("Layers1EtaFast");
  } else if ( iter == 2 ) { 
    layersFullP = (TH2F*) gDirectory->Get("Layers2PFull");
    layersFastP = (TH2F*) gDirectory->Get("Layers2PFast");
    layersFullEta = (TH2F*) gDirectory->Get("Layers2EtaFull");
    layersFastEta = (TH2F*) gDirectory->Get("Layers2EtaFast");
  } else if ( iter == 3 ) { 
    layersFullP = (TH2F*) gDirectory->Get("Layers3PFull");
    layersFastP = (TH2F*) gDirectory->Get("Layers3PFast");
    layersFullEta = (TH2F*) gDirectory->Get("Layers3EtaFull");
    layersFastEta = (TH2F*) gDirectory->Get("Layers3EtaFast");
  } else if ( iter == 4 ) { 
    layersFullP = (TH2F*) gDirectory->Get("Layers4PFull");
    layersFastP = (TH2F*) gDirectory->Get("Layers4PFast");
    layersFullEta = (TH2F*) gDirectory->Get("Layers4EtaFull");
    layersFastEta = (TH2F*) gDirectory->Get("Layers4EtaFast");
  } else if ( iter == 5 ) { 
    layersFullP = (TH2F*) gDirectory->Get("Layers5PFull");
    layersFastP = (TH2F*) gDirectory->Get("Layers5PFast");
    layersFullEta = (TH2F*) gDirectory->Get("Layers5EtaFull");
    layersFastEta = (TH2F*) gDirectory->Get("Layers5EtaFast");
  }
  for (unsigned imom=1;imom<6;++imom) {  
    layersFullP->ProjectionY(titleFull[imom-1],20*(imom-1)+1,20*imom+1);
    layersFastP->ProjectionY(titleFast[imom-1],20*(imom-1)+1,20*imom+1);
    layersFullEta->ProjectionY(titleFull[10-imom],3*(imom-1),max((unsigned int)28,30-3*(imom-1)));
    layersFastEta->ProjectionY(titleFast[10-imom],3*(imom-1),max((unsigned int)28,30-3*(imom-1)));
    layersFullEta->ProjectionY("cacaFull",3*imom,30-3*imom);
    layersFastEta->ProjectionY("cacaFast",3*imom,30-3*imom);
    TH1F* fastP = (TH1F*) gDirectory->Get(titleFast[imom-1]);
    ,    TH1F* fullP = (TH1F*) gDirectory->Get(titleFull[imom-1]);
    TH1F* fastEta = (TH1F*) gDirectory->Get(titleFast[10-imom]);
    TH1F* fullEta = (TH1F*) gDirectory->Get(titleFull[10-imom]);
    TH1F* fastEta2 = (TH1F*) gDirectory->Get("cacaFast");
    TH1F* fullEta2 = (TH1F*) gDirectory->Get("cacaFull");
    fastEta->Add(fastEta2,-1);
    fullEta->Add(fullEta2,-1);
    
    c->cd(imom);
    PlotGraphs(fullP,fastP);
    c->cd(12-imom);
    PlotGraphs(fullEta,fastEta);
  }  
  
}


void Seed(unsigned int iter) {
  
  gROOT->Reset();
    TFile *f = new TFile("testTrackingIterations.root");
    //TFile *f = new TFile("testTrackingIterations_fullstat.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  std::vector<char *> titleFull;
  std::vector<char *> titleFast;
  titleFull.push_back("SeedFull_p_0_2");
  titleFull.push_back("SeedFull_p_2_4");
  titleFull.push_back("SeedFull_p_4_6");
  titleFull.push_back("SeedFull_p_6_8");
  titleFull.push_back("SeedFull_p_8_10");
  titleFull.push_back("SeedFull_Eta_0.0_0.5");
  titleFull.push_back("SeedFull_Eta_0.5_1.0");
  titleFull.push_back("SeedFull_Eta_1.0_1.5");
  titleFull.push_back("SeedFull_Eta_1.5_2.0");
  titleFull.push_back("SeedFull_Eta_2.0_2.8");
  titleFast.push_back("SeedFast_p_0_2");
  titleFast.push_back("SeedFast_p_2_4");
  titleFast.push_back("SeedFast_p_4_6");
  titleFast.push_back("SeedFast_p_6_8");
  titleFast.push_back("SeedFast_p_8_10");
  titleFast.push_back("SeedFast_Eta_0.0_0.6");
  titleFast.push_back("SeedFast_Eta_0.6_1.2");
  titleFast.push_back("SeedFast_Eta_1.2_1.8");
  titleFast.push_back("SeedFast_Eta_1.8_2.4");
  titleFast.push_back("SeedFast_Eta_2.4_3.0");

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(3,4);
  TH2F* seedFullP;
  TH2F* seedFastP;
  TH2F* seedFullEta;
  TH2F* seedFastEta;
  if ( iter == 3 ) { 
    seedFullP = (TH2F*) gDirectory->Get("Seed3PFull");
    seedFastP = (TH2F*) gDirectory->Get("Seed3PFast");
    seedFullEta = (TH2F*) gDirectory->Get("Seed3EtaFull");
    seedFastEta = (TH2F*) gDirectory->Get("Seed3EtaFast");
  }
  else   if ( iter == 5 ) { 
    seedFullP = (TH2F*) gDirectory->Get("Seed5PFull");
    seedFastP = (TH2F*) gDirectory->Get("Seed5PFast");
    seedFullEta = (TH2F*) gDirectory->Get("Seed5EtaFull");
    seedFastEta = (TH2F*) gDirectory->Get("Seed5EtaFast");
  }
  for (unsigned imom=1;imom<6;++imom) {  
    seedFullP->ProjectionY(titleFull[imom-1],20*(imom-1)+1,20*imom+1);
    seedFastP->ProjectionY(titleFast[imom-1],20*(imom-1)+1,20*imom+1);
    seedFullEta->ProjectionY(titleFull[10-imom],3*(imom-1),max((unsigned int)28,30-3*(imom-1)));
    seedFastEta->ProjectionY(titleFast[10-imom],3*(imom-1),max((unsigned int)28,30-3*(imom-1)));
    seedFullEta->ProjectionY("cacaFull",3*imom,30-3*imom);
    seedFastEta->ProjectionY("cacaFast",3*imom,30-3*imom);

    TH1F* fastP = (TH1F*) gDirectory->Get(titleFast[imom-1]);
    TH1F* fullP = (TH1F*) gDirectory->Get(titleFull[imom-1]);
    TH1F* fastEta = (TH1F*) gDirectory->Get(titleFast[10-imom]);
    TH1F* fullEta = (TH1F*) gDirectory->Get(titleFull[10-imom]);
    TH1F* fastEta2 = (TH1F*) gDirectory->Get("cacaFast");
    TH1F* fullEta2 = (TH1F*) gDirectory->Get("cacaFull");
    fastEta->Add(fastEta2,-1);
    fullEta->Add(fullEta2,-1);
    
    c->cd(imom);
    PlotGraphs(fullP,fastP);
    c->cd(12-imom);
    PlotGraphs(fullEta,fastEta);
  }  
  
}



void totalEfficiency(unsigned int iter) {
  
  gROOT->Reset();

    TFile *f = new TFile("testTrackingIterations.root");
  //TFile *f = new TFile("testTrackingIterations_fullstat.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(1,2);
  TH2F* iter0Fast;
  TH2F* iter1Fast;
  TH2F* iter2Fast;
  TH2F* iter3Fast;
  TH2F* iter4Fast;
  TH2F* iter5Fast;
  TH2F* iter0Full;
  TH2F* iter1Full;
  TH2F* iter2Full;
  TH2F* iter3Full;
  TH2F* iter4Full;
  TH2F* iter5Full;
  TH2F* genPlot;
  genPlot = (TH2F*) gDirectory->Get("genEtaP");
  iter0Full = (TH2F*) gDirectory->Get("eff0Full");
  iter0Fast = (TH2F*) gDirectory->Get("eff0Fast");
  iter1Full = (TH2F*) gDirectory->Get("eff1Full");
  iter1Fast = (TH2F*) gDirectory->Get("eff1Fast");
  iter2Full = (TH2F*) gDirectory->Get("eff2Full");
  iter2Fast = (TH2F*) gDirectory->Get("eff2Fast");
  iter3Full = (TH2F*) gDirectory->Get("eff3Full");
  iter3Fast = (TH2F*) gDirectory->Get("eff3Fast");
  iter4Full = (TH2F*) gDirectory->Get("eff4Full");
  iter4Fast = (TH2F*) gDirectory->Get("eff4Fast");
  iter5Full = (TH2F*) gDirectory->Get("eff5Full");
  iter5Fast = (TH2F*) gDirectory->Get("eff5Fast");
  
  TH1F* fast0;
  TH1F* full0;
  TH1F* fast1;
  TH1F* full1;
  TH1F* fast2;
  TH1F* full2;
  TH1F* fast3;
  TH1F* full3;
  TH1F* fast4;
  TH1F* full4;
  TH1F* fast5;
  TH1F* full5;
  TH1F* gen;
  TH1F* fastp0;
  TH1F* fullp0;
  TH1F* fastp1;
  TH1F* fullp1;
  TH1F* fastp2;
  TH1F* fullp2;
  TH1F* fastp3;
  TH1F* fullp3;
  TH1F* fastp4;
  TH1F* fullp4;
  TH1F* fastp5;
  TH1F* fullp5;
  TH1F* genp;
  genPlot->ProjectionX();
  iter0Full->ProjectionX("iter0Full_px");
  iter0Fast->ProjectionX("iter0Fast_px");
  iter1Full->ProjectionX("iter1Full_px");
  iter1Fast->ProjectionX("iter1Fast_px");
  iter2Full->ProjectionX("iter2Full_px");
  iter2Fast->ProjectionX("iter2Fast_px");
  iter3Full->ProjectionX("iter3Full_px");
  iter3Fast->ProjectionX("iter3Fast_px");
  iter4Full->ProjectionX("iter4Full_px");
  iter4Fast->ProjectionX("iter4Fast_px");
  iter5Full->ProjectionX("iter5Full_px");
  iter5Fast->ProjectionX("iter5Fast_px");
  genPlot->ProjectionY();
  iter0Full->ProjectionY("iter0Full_py");
  iter0Fast->ProjectionY("iter0Fast_py");
  iter1Full->ProjectionY("iter1Full_py");
  iter1Fast->ProjectionY("iter1Fast_py");
  iter2Full->ProjectionY("iter2Full_py");
  iter2Fast->ProjectionY("iter2Fast_py");
  iter3Full->ProjectionY("iter3Full_py");
  iter3Fast->ProjectionY("iter3Fast_py");
  iter4Full->ProjectionY("iter4Full_py");
  iter4Fast->ProjectionY("iter4Fast_py");
  iter5Full->ProjectionY("iter5Full_py");
  iter5Fast->ProjectionY("iter5Fast_py");
  fast0  = (TH1F*) gDirectory->Get("iter0Fast_px");
  full0  = (TH1F*) gDirectory->Get("iter0Full_px");
  fastp0 = (TH1F*) gDirectory->Get("iter0Fast_py");
  fullp0 = (TH1F*) gDirectory->Get("iter0Full_py");
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
  fast4  = (TH1F*) gDirectory->Get("iter4Fast_px");
  full4  = (TH1F*) gDirectory->Get("iter4Full_px");
  fastp4 = (TH1F*) gDirectory->Get("iter4Fast_py");
  fullp4 = (TH1F*) gDirectory->Get("iter4Full_py");
  fast5  = (TH1F*) gDirectory->Get("iter5Fast_px");
  full5  = (TH1F*) gDirectory->Get("iter5Full_px");
  fastp5 = (TH1F*) gDirectory->Get("iter5Fast_py");
  fullp5 = (TH1F*) gDirectory->Get("iter5Full_py");
  gen = (TH1F*) gDirectory->Get("genEtaP_px");
  genp = (TH1F*) gDirectory->Get("genEtaP_py");

  if ( iter == 0 ) fast1 = fast0;
  if ( iter == 2 ) fast1 = fast2;
  if ( iter == 3 ) fast1 = fast3;
  if ( iter == 4 ) fast1 = fast4;
  if ( iter == 5 ) fast1 = fast5;
  if ( iter > 11 ) fast1->Add(fast2);
  if ( iter > 12 ) fast1->Add(fast3);
  fast1->Divide(gen);
  if ( iter == 0 ) full1 = full0;
  if ( iter == 2 ) full1 = full2;
  if ( iter == 3 ) full1 = full3;
  if ( iter == 4 ) full1 = full4;
  if ( iter == 5 ) full1 = full5;
  if ( iter > 11 ) full1->Add(full2);
  if ( iter > 12 ) full1->Add(full3);
  full1->Divide(gen);
  c->cd(1);
  PlotGraphs(full1,fast1);

  if ( iter == 0 ) fastp1 = fastp0;
  if ( iter == 2 ) fastp1 = fastp2;
  if ( iter == 3 ) fastp1 = fastp3;
  if ( iter == 4 ) fastp1 = fastp4;
  if ( iter == 5 ) fastp1 = fastp5;
  if ( iter > 11 ) fastp1->Add(fastp2);
  if ( iter > 12 ) fastp1->Add(fastp3);
  fastp1->Divide(genp);
  if ( iter == 0 ) fullp1 = fullp0;
  if ( iter == 2 ) fullp1 = fullp2;
  if ( iter == 3 ) fullp1 = fullp3;
  if ( iter == 4 ) fullp1 = fullp4;
  if ( iter == 5 ) fullp1 = fullp5;
  if ( iter > 11 ) fullp1->Add(fullp2);
  if ( iter > 12 ) fullp1->Add(fullp3);
  fullp1->Divide(genp);
  c->cd(2);
  PlotGraphs(fullp1,fastp1);

}

void SimTracks() {
  
  gROOT->Reset();
  TFile *f = new TFile("testTrackingIterations.root");
  //  TFile *f = new TFile("testTrackingIterations_fullstat.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  std::vector<char *> titleFull;
  std::vector<char *> titleFast;
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

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(3,4);
  TH2F* effFull;
  TH2F* effFast;
  effFull = (TH2F*) gDirectory->Get("SimFull");
  effFast = (TH2F*) gDirectory->Get("SimFast");
  
  for (unsigned imom=1;imom<11;++imom) {  
    effFull->ProjectionX(titleFull[imom-1],10*(imom-1)+1,10*imom+1);
    effFast->ProjectionX(titleFast[imom-1],10*(imom-1)+1,10*imom+1);
    TH1F* fast = (TH1F*) gDirectory->Get(titleFast[imom-1]);
    TH1F* full = (TH1F*) gDirectory->Get(titleFull[imom-1]);
    c->cd(imom);
    PlotGraphs(full,fast);
  }  

  
  TH1F* fast;
  TH1F* full;
  TH1F* fastp;
  TH1F* fullp;
  effFull->ProjectionX();
  effFast->ProjectionX();
  effFull->ProjectionY();
  effFast->ProjectionY();
  
  fast = (TH1F*) gDirectory->Get("SimFast_px");
  full = (TH1F*) gDirectory->Get("SimFull_px");
  fastp = (TH1F*) gDirectory->Get("SimFast_py");
  fullp = (TH1F*) gDirectory->Get("SimFull_py");
  
  c->cd(11);
  PlotGraphs(full,fast);
  
  c->cd(12);
  PlotGraphs(fullp,fastp);

}
