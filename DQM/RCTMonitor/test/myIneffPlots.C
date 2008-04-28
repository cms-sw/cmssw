{
  Float_t greenRange  = .05;
  Float_t blueRange   = .1;
  Float_t yellowRange = .15;
  Float_t orangeRange = .2;

  TCanvas *c1 = new TCanvas();

  TFile *f = new TFile("myTestFile.root");

  TH2I *dqmElecIneff2d   = new TH2I("dqmElecIneff2d",   "2D inefficiency for electrons",                           8, 6.5, 14.5, 18, -0.5, 17.5);
  TH2I *dqmElecOvereff2d = new TH2I("dqmElecOvereff2d", "2D overefficiency for electrons",                         8, 6.5, 14.5, 18, -0.5, 17.5);
  TH2I *dqmMatched       = new TH2I("dqmMatched",       "Inefficiency for matched electrons with nonzero Delta E", 8, 6.5, 14.5, 18, -0.5, 17.5);

  TH2F *boxes = new TH2F("boxes", "", 8, 6.5, 14.5, 18, -0.5, 17.5);

  for(Int_t i = 7; i < 15; i++)
  {
    for(Int_t j = 3;  j < 7;  j++) boxes->Fill(i, j, 10);
    for(Int_t j = 12; j < 16; j++) boxes->Fill(i, j, 10);
  }

  for(Int_t i = 1; i < 9; i++)
  {
    for(Int_t j = 4; j < 8; j++)
    {
      if(elecIneff2d           ->GetBinContent(i, j) < 0.01) elecIneff2d           ->Fill(i + 6, j - 1, 0.01);
      if(elecOvereff2d         ->GetBinContent(i, j) < 0.01) elecOvereff2d         ->Fill(i + 6, j - 1, 0.01);
      if(matchedElecDeltaEIneff->GetBinContent(i, j) < 0.01) matchedElecDeltaEIneff->Fill(i + 6, j - 1, 0.01);
    }

    for(Int_t j = 13; j < 17; j++)
    {
      if(elecIneff2d           ->GetBinContent(i, j) < 0.01) elecIneff2d           ->Fill(i + 6, j - 1, 0.01);
      if(elecOvereff2d         ->GetBinContent(i, j) < 0.01) elecOvereff2d         ->Fill(i + 6, j - 1, 0.01);
      if(matchedElecDeltaEIneff->GetBinContent(i, j) < 0.01) matchedElecDeltaEIneff->Fill(i + 6, j - 1, 0.01);
    }
  }

  for(Int_t i = 1; i < 9; i++)
    for(Int_t j = 1; j < 19; j++)
    {
      if(elecIneff2d->GetBinContent(i, j) >  0.          && elecIneff2d->GetBinContent(i, j) <  greenRange)  dqmElecIneff2d->Fill(i + 6, j - 1, 5);
      if(elecIneff2d->GetBinContent(i, j) >= greenRange  && elecIneff2d->GetBinContent(i, j) <  blueRange)   dqmElecIneff2d->Fill(i + 6, j - 1, 4);
      if(elecIneff2d->GetBinContent(i, j) >= blueRange   && elecIneff2d->GetBinContent(i, j) <  yellowRange) dqmElecIneff2d->Fill(i + 6, j - 1, 3);
      if(elecIneff2d->GetBinContent(i, j) >= yellowRange && elecIneff2d->GetBinContent(i, j) <  orangeRange) dqmElecIneff2d->Fill(i + 6, j - 1, 2);
      if(elecIneff2d->GetBinContent(i, j) >= orangeRange && elecIneff2d->GetBinContent(i, j) <= 1.)          dqmElecIneff2d->Fill(i + 6, j - 1, 1);

      if(elecOvereff2d->GetBinContent(i, j) >  0.          && elecOvereff2d->GetBinContent(i, j) <  greenRange)  dqmElecOvereff2d->Fill(i + 6, j - 1, 5);
      if(elecOvereff2d->GetBinContent(i, j) >= greenRange  && elecOvereff2d->GetBinContent(i, j) <  blueRange)   dqmElecOvereff2d->Fill(i + 6, j - 1, 4);
      if(elecOvereff2d->GetBinContent(i, j) >= blueRange   && elecOvereff2d->GetBinContent(i, j) <  yellowRange) dqmElecOvereff2d->Fill(i + 6, j - 1, 3);
      if(elecOvereff2d->GetBinContent(i, j) >= yellowRange && elecOvereff2d->GetBinContent(i, j) <  orangeRange) dqmElecOvereff2d->Fill(i + 6, j - 1, 2);
      if(elecOvereff2d->GetBinContent(i, j) >= orangeRange && elecOvereff2d->GetBinContent(i, j) <= 1.)          dqmElecOvereff2d->Fill(i + 6, j - 1, 1);

      if(matchedElecDeltaEIneff->GetBinContent(i, j) >  0.          && matchedElecDeltaEIneff->GetBinContent(i, j) <  greenRange)  dqmMatched->Fill(i + 6, j - 1, 5);
      if(matchedElecDeltaEIneff->GetBinContent(i, j) >= greenRange  && matchedElecDeltaEIneff->GetBinContent(i, j) <  blueRange)   dqmMatched->Fill(i + 6, j - 1, 4);
      if(matchedElecDeltaEIneff->GetBinContent(i, j) >= blueRange   && matchedElecDeltaEIneff->GetBinContent(i, j) <  yellowRange) dqmMatched->Fill(i + 6, j - 1, 3);
      if(matchedElecDeltaEIneff->GetBinContent(i, j) >= yellowRange && matchedElecDeltaEIneff->GetBinContent(i, j) <  orangeRange) dqmMatched->Fill(i + 6, j - 1, 2);
      if(matchedElecDeltaEIneff->GetBinContent(i, j) >= orangeRange && matchedElecDeltaEIneff->GetBinContent(i, j) <= 1.)          dqmMatched->Fill(i + 6, j - 1, 1);
    }

  c1->Clear();

  elecIneff1d->Draw();
  elecIneff1d->Draw("HIST SAME");
  elecIneff1d->GetXaxis()->SetTitle("channel (#eta + 8#phi)");
  elecIneff1d->GetYaxis()->SetTitle("inefficiency");
  elecIneff1d->GetYaxis()->SetRangeUser(0., 1.1);
  c1->SaveAs("./myPlots/elecIneff1d.png");

  c1->Clear();

  elecOvereff1d->Draw();
  elecOvereff1d->Draw("HIST SAME");
  elecOvereff1d->GetXaxis()->SetTitle("channel (#eta + 8#phi)");
  elecOvereff1d->GetYaxis()->SetTitle("overefficiency");
  // elecOvereff1d->GetYaxis()->SetRangeUser(0., 1.1);
  c1->SaveAs("./myPlots/elecOvereff1d.png");

  c1->Clear();

  dqmElecIneff2d->Draw("COL");
  boxes->Draw("SAME BOX");
  dqmElecIneff2d->GetXaxis()->SetTitle("channel (#eta)");
  dqmElecIneff2d->GetYaxis()->SetTitle("channel (#phi)");
  dqmElecIneff2d->SetMaximum(6.);
  c1->SaveAs("./myPlots/elecIneff2d.png");

  c1->Clear();

  dqmElecOvereff2d->Draw("COL");
  boxes->Draw("SAME BOX");
  dqmElecOvereff2d->GetXaxis()->SetTitle("channel (#eta)");
  dqmElecOvereff2d->GetYaxis()->SetTitle("channel (#phi)");
  dqmElecOvereff2d->SetMaximum(6.);
  c1->SaveAs("./myPlots/elecOvereff2d.png");

  c1->Clear();

  dqmMatched->Draw("COL");
  boxes->Draw("SAME BOX");
  dqmMatched->GetXaxis()->SetTitle("channel (#eta)");
  dqmMatched->GetYaxis()->SetTitle("channel (#phi)");
  dqmMatched->SetMaximum(6.);
  c1->SaveAs("./myPlots/matchedElecDeltaEIneff.png");
}
