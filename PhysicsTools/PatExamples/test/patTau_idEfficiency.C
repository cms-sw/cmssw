void showEfficiency(TCanvas& canvas, TH1* hTauJetPtDiscrPassed, const char* paramName, const char* xAxisLabel)
{
//--- auxiliary function to show a single efficiency plot
//    and save the result in an .eps file

  TString plotTitle = TString("Tau Id. Efficiency as function of").Append(" ").Append(paramName);
  hTauJetPtDiscrPassed->SetTitle(plotTitle.Data());

  hTauJetPtDiscrPassed->SetMinimum(0.);
  hTauJetPtDiscrPassed->SetMaximum(1.2);

  hTauJetPtDiscrPassed->SetXTitle(paramName);
  hTauJetPtDiscrPassed->SetYTitle("#varepsilon");

  hTauJetPtDiscrPassed->Draw("e1p");

  canvas.Update();

  TString outputFileName = TString("tauIdEff").Append("_").Append(paramName).Append(".eps");
  canvas.Print(outputFileName.Data());
}

void patTau_idEfficiency()
{
//--- open ROOT file
  TFile inputFile("patTau_Histograms.root");

//--- descend into directory containing histograms 
//    written by TFileService
  TDirectory* inputDir = (TDirectory*)inputFile.Get("analyzePatTau");

  TH1* hTauJetPt = (TH1*)inputDir->Get("TauJetPt");
  TH1* hTauJetEta = (TH1*)inputDir->Get("TauJetEta");
  TH1* hTauJetPhi = (TH1*)inputDir->Get("TauJetPhi");

  TH1* hTauJetPtDiscrPassed = (TH1*)inputDir->Get("TauJetPtIsoPassed");
  TH1* hTauJetEtaDiscrPassed = (TH1*)inputDir->Get("TauJetEtaIsoPassed");
  TH1* hTauJetPhiDiscrPassed = (TH1*)inputDir->Get("TauJetPhiIsoPassed");

//--- compute tau id. efficiency as function of Pt, Eta and Phi
//    by dividing histograms;
//    enable computation of uncertainties (binomial errors) 
//    by calling TH1::Sumw2 before TH1::Divide
  hTauJetPtDiscrPassed->Sumw2();
  hTauJetPtDiscrPassed->Divide(hTauJetPtDiscrPassed, hTauJetPt, 1., 1., "B");
  hTauJetEtaDiscrPassed->Sumw2();
  hTauJetEtaDiscrPassed->Divide(hTauJetEtaDiscrPassed, hTauJetEta, 1., 1., "B");
  hTauJetPhiDiscrPassed->Sumw2();
  hTauJetPhiDiscrPassed->Divide(hTauJetPhiDiscrPassed, hTauJetPhi, 1., 1., "B");

//--- create canvas on which to draw the plots;
//    switch background color from (light)grey to white
  TCanvas canvas("plotPatTauEff", "plotPatTauEff", 800, 600);
  canvas.SetFillColor(10);

//--- draw efficiciency plots
  showEfficiency(canvas, hTauJetPtDiscrPassed, "Pt", "P_{T} / GeV");
  showEfficiency(canvas, hTauJetEtaDiscrPassed, "Eta", "#eta");
  showEfficiency(canvas, hTauJetPhiDiscrPassed, "Phi", "#phi");

//--- close ROOT file
  inputFile.Close();
}
