void drawValidationPlot() {
  TFile* f = new TFile("DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root");
  TDirectoryFile* df = f->Get<TDirectoryFile>("DQMData")
                           ->Get<TDirectoryFile>("Run 1")
                           ->Get<TDirectoryFile>("ParticleFlow")
                           ->Get<TDirectoryFile>("Run summary")
                           ->Get<TDirectoryFile>("PFRecHitV");

  TCanvas* c = new TCanvas();
  c->SetLogz();

  df->Get<TH2>("energy")->Draw("COLZ");
  c->SaveAs("pfRecHit_energy.png");
  c->Clear();

  df->Get<TH2>("time")->Draw("COLZ");
  c->SaveAs("pfRecHit_time.png");
}
