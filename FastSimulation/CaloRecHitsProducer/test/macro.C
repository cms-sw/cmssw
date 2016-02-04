{
  TFile * fast = new TFile("Noisecheck-fast.root");
  TFile * full = new TFile("Noisecheck-full.root");
  full->cd("DQMData");
  EEPP->SetLineColor(kRed);
  EEPP->Draw();
  EEPN->SetLineColor(kRed+2);
  EEPN->Draw("same");
  EENN->SetLineColor(kBlue);
  EENN->Draw("same");
  EENP->SetLineColor(kBlue+2);
  EENP->Draw("same");
}
