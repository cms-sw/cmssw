void TestCorrections()
{
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1);
  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable(); 
  TRandom *rnd = new TRandom();
  rnd->SetSeed(0);
  ////////////// Define the JetCorrector ///////////////////////
  string Levels = "L4";
  string Tags = "../data/L4EMF_AK5Calo.txt:";
  FactorizedJetCorrector *JetCorrector = new FactorizedJetCorrector(Levels,Tags);
  ////////////// Fill demo histogram  ////////////////////////////
  double vx[1000],vy[1000];
  for(unsigned index = 0; index < 1000; index++ ) 
    {
      double pt  = rnd->Uniform(10,500);    
      double eta = rnd->Uniform(-5,5);  
      double emf = rnd->Uniform(-1,1);
      JetCorrector->setJetEta(eta); 
      JetCorrector->setJetPt(pt);
      JetCorrector->setJetEMF(emf);  
      double scale = JetCorrector->getCorrection();
      vx[index] = pt;
      vy[index] = scale;
    }
  TGraph *g = new TGraph(1000,vx,vy);
  ////////////// Draw demo histograms /////////////////////////////
  TCanvas *c = new TCanvas("Correction","Correction");
  g->SetTitle("Correction");
  g->GetXaxis()->SetTitle("p_{T} (GeV)");
  g->GetYaxis()->SetTitle("Correction Factor");
  g->Draw("AP"); 
}
