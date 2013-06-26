#include <vector>

void PlotGraphs( TH1F* full, TH1F* fast) {

  full->SetMarkerColor(4);						
  full->SetLineColor(4);						  
  full->SetLineWidth(2);						  
  full->SetLineStyle(3);
  //  full->SetMaximum(0.11);
  full->Draw();


  fast->SetMarkerStyle(25);						
  fast->SetMarkerColor(2);						
  fast->SetLineColor(2);						  
  fast->SetLineWidth(2);						  
  fast->Draw("same");
  
}

void Efficiency(unsigned int iter) {
  
  gROOT->Reset();
  TFile *f = new TFile("testGeneralTracks.root");
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
    TFile *f = new TFile("testGeneralTracks.root");
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
    TFile *f = new TFile("testGeneralTracks.root");
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

    TFile *f = new TFile("testGeneralTracks.root");
  //TFile *f = new TFile("testTrackingIterations_fullstat.root");
  if(f->IsZombie() ) return;
  f->cd("DQMData"); 

  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(1,2);
  TH2F* iter0Fast;
  TH2F* iter0Full;
  TH2F* genPlot;
  genPlot = (TH2F*) gDirectory->Get("genEtaP");
  iter0Full = (TH2F*) gDirectory->Get("eff0Full");
  iter0Fast = (TH2F*) gDirectory->Get("eff0Fast");

  TH1F* fast0;
  TH1F* full0;
  TH1F* gen;
  TH1F* fastp0;
  TH1F* fullp0;
  TH1F* genp;
  genPlot->ProjectionX();
  iter0Full->ProjectionX("iter0Full_px");
  iter0Fast->ProjectionX("iter0Fast_px");
  genPlot->ProjectionY();
  iter0Full->ProjectionY("iter0Full_py");
  iter0Fast->ProjectionY("iter0Fast_py");
  fast0  = (TH1F*) gDirectory->Get("iter0Fast_px");
  full0  = (TH1F*) gDirectory->Get("iter0Full_px");
  fastp0 = (TH1F*) gDirectory->Get("iter0Fast_py");
  fullp0 = (TH1F*) gDirectory->Get("iter0Full_py");
  gen = (TH1F*) gDirectory->Get("genEtaP_px");
  genp = (TH1F*) gDirectory->Get("genEtaP_py");


  fast0->Divide(gen);
  full0->Divide(gen);
  c->cd(1);
  PlotGraphs(full0,fast0);

  fastp0->Divide(genp);
  fullp0->Divide(genp);
  c->cd(2);
  PlotGraphs(fullp0,fastp0);

}

