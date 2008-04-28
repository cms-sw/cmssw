{
  TCanvas *c1 = new TCanvas();

  TFile *f = new TFile("myTestFile.root");

  c1->Clear();

  ecalTpOcc->Draw("BOX");
  ecalTpOcc->GetXaxis()->SetTitle("channel (#eta)");
  ecalTpOcc->GetYaxis()->SetTitle("channel (#phi)");
  c1->SaveAs("./myPlots/ecalTpOcc.png");

  c1->Clear();

  hcalTpOcc->Draw("BOX");
  hcalTpOcc->GetXaxis()->SetTitle("channel (#eta)");
  hcalTpOcc->GetYaxis()->SetTitle("channel (#phi)");
  c1->SaveAs("./myPlots/hcalTpOcc.png");

  c1->Clear();

  hwElecOcc->Draw("BOX");
  hwElecOcc->GetXaxis()->SetTitle("channel (#eta)");
  hwElecOcc->GetYaxis()->SetTitle("channel (#phi)");
  c1->SaveAs("./myPlots/hwElecOcc.png");

  c1->Clear();

  emulElecOcc2d->Draw("BOX");
  emulElecOcc2d->GetXaxis()->SetTitle("channel (#eta)");
  emulElecOcc2d->GetYaxis()->SetTitle("channel (#phi)");
  c1->SaveAs("./myPlots/emulElecOcc2d.png");

  c1->Clear();

  matchedElecDeltaE->Draw();
  matchedElecDeltaE->GetXaxis()->SetTitle("channel (#eta + 8#phi)");
  matchedElecDeltaE->GetYaxis()->SetTitle("E_{emul} - E_{HW}");
  c1->SaveAs("./myPlots/matchedElecDeltaE.png");
}
