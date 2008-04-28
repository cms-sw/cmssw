{
  Float_t redRange    = .8;
  Float_t orangeRange = .85;
  Float_t yellowRange = .9;
  Float_t blueRange   = .95;

  TCanvas *c1 = new TCanvas();

  TFile *f = new TFile("myTestFile.root");

  TH2I *dqmElecEff2d = new TH2I("dqmElecEff2d", "2D efficiency for electrons", 8, 6.5, 14.5, 18, -0.5, 17.5); 

  TH2F *boxes = new TH2F("boxes", "", 8, 6.5, 14.5, 18, -0.5, 17.5);

  for(Int_t i = 1; i < 9; i++)
    for(Int_t j = 1; j < 19; j++)
    {
      Double_t binCon = elecEff2d->GetBinContent(i, j);

      if(binCon >  0.          && binCon <  redRange)    dqmElecEff2d->Fill(i + 6, j - 1, 1);
      if(binCon >= redRange    && binCon <  orangeRange) dqmElecEff2d->Fill(i + 6, j - 1, 2);
      if(binCon >= orangeRange && binCon <  yellowRange) dqmElecEff2d->Fill(i + 6, j - 1, 3);
      if(binCon >= yellowRange && binCon <  blueRange)   dqmElecEff2d->Fill(i + 6, j - 1, 4);
      if(binCon >= blueRange   && binCon <= 1.)          dqmElecEff2d->Fill(i + 6, j - 1, 5);
    }

  for(Int_t i = 7; i < 15; i++)
  {
    for(Int_t j = 3;  j < 7;  j++) boxes->Fill(i, j, 10.);
    for(Int_t j = 12; j < 16; j++) boxes->Fill(i, j, 10.);
  }

  c1->Clear();

  elecEff1d->Draw();
  elecEff1d->Draw("HIST SAME");
  elecEff1d->GetXaxis()->SetTitle("channel (#eta + 8#phi)");
  elecEff1d->GetYaxis()->SetTitle("efficiency");
  elecEff1d->GetYaxis()->SetRangeUser(0.0, 1.1);
  c1->SaveAs("./myPlots/elecEff1d.png");

  c1->Clear();

  dqmElecEff2d->Draw("COL");
  boxes->Draw("SAME BOX");
  dqmElecEff2d->GetXaxis()->SetTitle("channel (#eta)");
  dqmElecEff2d->GetYaxis()->SetTitle("channel (#phi)");
  dqmElecEff2d->SetMaximum(6.);
  c1->SaveAs("./myPlots/elecEff2d.png");
}
