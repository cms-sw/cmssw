void draw(TH1D * h, TF1 * f) {
  int lineWidth = 0.4;
  Color_t lineColor = kRed;
  f->SetLineColor(lineColor);
  f->SetLineWidth(lineWidth);
  TCanvas c(h->GetName(),h->GetTitle(),1000,800);
  c.cd();
  h->Draw();
  f->Draw("same");

  // Text box with fit results
  fitLabel = new TPaveText(0.65,0.3,0.85,0.5,"NDC");
  fitLabel->SetBorderSize(1);
  fitLabel->SetTextAlign(12);
  fitLabel->SetTextSize(0.02);
  fitLabel->SetFillColor(0);
  fitLabel->AddText("Function: "+f->GetExpFormula());
  for( int i=0; i<f->GetNpar(); ++i ) {
    char name[50];
    cout << "par["<<i<<"] = " << f->GetParameter(i) << endl;
    sprintf(name, "par[%i] = %4.2g #pm %4.2g",i, f->GetParameter(i), f->GetParError(i));
    fitLabel->AddText(name);
  }
  fitLabel->Draw("same");

  c.Write();
  h->Write();
}

int ResolFit() {

  TFile inputFile("redrawed.root","READ");
  TFile outputFile("fitted.root","RECREATE");

  outputFile.cd();

  // Pt resolution
  // -------------
  // VS pt
  cout << "Fitting Pt resolution vs Pt" << endl;
  TDirectory * tempDir = (TDirectory*) inputFile.Get("hResolPtGenVSMu");
  TH1D * h = (TH1D*) tempDir->Get("hResolPtGenVSMu_ResoVSPt_resol");
  TF1 *f = new TF1("f","pol4",0,100);
  h->Fit("f","R0");
  h->SetMinimum(0);
  h->SetMaximum(0.1);
  draw(h,f);

//   TCanvas c("canvas", "canvas", 1000, 800);
//   c->cd();
//   h->Draw();

  // VS eta
  cout << "Fitting Pt resolution vs Eta" << endl;
  tempDir = (TDirectory*) inputFile.Get("hResolPtGenVSMu");
  h = (TH1D*) tempDir->Get("hResolPtGenVSMu_ResoVSEta_resol");
  f = new TF1("f","pol6",-2.5,2.5);
  h->Fit("f","R0");
  h->SetMinimum(0);
  h->SetMaximum(0.045);
  draw(h,f);

  // CotgTheta resolution
  // --------------------
  // VS pt
  cout << "Fitting CotgTheta resolution vs Pt" << endl;
  tempDir = (TDirectory*) inputFile.Get("hResolCotgThetaGenVSMu");
  h = (TH1D*) tempDir->Get("hResolCotgThetaGenVSMu_ResoVSPt_resol");
  f = new TF1("f","[0]/x+[1]",0,100);
  h->Fit("f","R0");
  h->SetMinimum(0);
  h->SetMaximum(0.015);
  draw(h,f);
  // VS eta
  cout << "Fitting CotgTheta resolution vs Eta" << endl;
  tempDir = (TDirectory*) inputFile.Get("hResolCotgThetaGenVSMu");
  h = (TH1D*) tempDir->Get("hResolCotgThetaGenVSMu_ResoVSEta_resol");
  f = new TF1("f","pol4",-3,3);
  h->Fit("f","R0");
  h->SetMinimum(0);
  h->SetMaximum(0.005);
  draw(h,f);

  // Phi resolution
  // --------------
  // VS pt
  cout << "Fitting Phi resolution vs Pt" << endl;
  tempDir = (TDirectory*) inputFile.Get("hResolPhiGenVSMu");
  h = (TH1D*) tempDir->Get("hResolPhiGenVSMu_ResoVSPt_resol");
  f = new TF1("f","[0]/x+[1]",0,100);
  h->Fit("f","R0");
  h->SetMinimum(0);
  h->SetMaximum(0.003);
  draw(h,f);
  // VS eta
  cout << "Fitting Phi resolution vs Eta" << endl;
  tempDir = (TDirectory*) inputFile.Get("hResolPhiGenVSMu");
  h = (TH1D*) tempDir->Get("hResolPhiGenVSMu_ResoVSEta_resol");
  f = new TF1("f","pol2",-2.4,2.4);
  h->Fit("f","R0");
  h->SetMinimum(0);
  h->SetMaximum(0.0005);
  draw(h,f);

  return 0;
}
