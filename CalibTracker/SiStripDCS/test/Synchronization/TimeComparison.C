/**
 * This macro can run on the output root file from SyncO2O and produce
 * a histogram showing the digis, digis with masking and number of
 * detIds with HV off. <br>
 * The digis with masking should remain always at a stable level during
 * a run and go to zero when the tracker HV is off. <br>
 * In contrast the digis will go up when the HV is being turned off. <br>
 * The increase in modules with HV off (provided a sufficient reduction
 * is used) will be a straigth line going up and will begin when the
 * off command is given to the modules from DCS.
 */

{
  TFile * inputFile = new TFile("digisAndHVvsTime.root", "read");
  TH1F * HVoff = (TH1F*)inputFile->Get("HVoff");
  TH1F * digisWithMasking = (TH1F*)inputFile->Get("digisWithMasking");
  TH1F * digis = (TH1F*)inputFile->Get("digis");

  TCanvas * canvas = new TCanvas("digiHVcomp", "digis vs HV off comparison", 1000, 800);
  canvas->Draw();

  digis->Draw();
  digisWithMasking->Draw("same");
  digisWithMasking->SetLineColor(kBlue);
  HVoff->Draw("same");
  HVoff->Scale(100);
  HVoff->SetLineColor(kRed);

  TLegend * legend = new TLegend(0.1, 0.7, 0.48, 0.9);
  legend->SetHeader("digis vs HV off comparison");
  legend->AddEntry(digis, "all digis");
  legend->AddEntry(digisWithMasking, "digis with HV off masking");
  legend->AddEntry(HVoff, "number of detId with HV off");
  legend->Draw("same");
}
