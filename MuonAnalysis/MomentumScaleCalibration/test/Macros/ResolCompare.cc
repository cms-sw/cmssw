// #include "TFile.h"
// #include "TDirectory.h"
// #include "TCanvas.h"
// #include "TH1D.h"
// #include "TProfile.h"

/**
 * This function draws the histograms superimposed in the same canvas. The second histogram is red.
 * Name of the resol histogram, name of the function resolution histogram and name of the canvas
 */
draw( const TString & resolName, TDirectory * resolDir,
      const TString & functionResolName, TDirectory * functionResolDir,
      const TString & canvasName, TFile * outputFile ) {
  TH1D * resolVSpt = (TH1D*) resolDir->Get(resolName);
  TProfile * functionResolVSpt = (TProfile*) functionResolDir->Get(functionResolName);

  TCanvas * c = new TCanvas(canvasName, canvasName, 1000, 800);
  c->cd();
  resolVSpt->Draw();
  functionResolVSpt->SetMarkerColor(kRed);
  functionResolVSpt->SetLineColor(kRed);
  functionResolVSpt->Draw("same");
  // c->Draw();
  outputFile->cd();
  c->Write();
}

/**
 * This macro compares the resolutions found with the resolution functions with those
 * obtained from recMu-genMu comparison in MuScleFit.
 * The true resolutions are writted in the "redrawed.root" file by the ResolDraw.cc macro.
 */
void ResolCompare() {

  TFile * outputFile = new TFile("ComparedResol.root", "RECREATE");

  TFile * resolFile = new TFile("redrawed.root", "READ");
  TFile * functionResolFile = new TFile("0_MuScleFit.root", "READ");

  TDirectory * resolDir = 0;
  TDirectory * functionResolDir = 0;
  // sigmaPt
  // -------
  resolDir = (TDirectory*) resolFile->Get("hResolPtGenVSMu");
  functionResolDir = (TDirectory*) functionResolFile->Get("hFunctionResolPt");
  // VS Pt
  draw("hResolPtGenVSMu_ResoVSPt_resol", resolDir,
       "hFunctionResolPt_ResoVSPt_prof", functionResolDir,
       "resolPtVSpt", outputFile);
  // VS Eta
  draw("hResolPtGenVSMu_ResoVSEta_resol", resolDir,
       "hFunctionResolPt_ResoVSEta_prof", functionResolDir,
       "resolPtVSeta", outputFile);

  // sigmaCotgTheta
  // --------------
  resolDir = (TDirectory*) resolFile->Get("hResolCotgThetaGenVSMu");
  functionResolDir = (TDirectory*) functionResolFile->Get("hFunctionResolCotgTheta");
  // VS Pt
  draw("hResolCotgThetaGenVSMu_ResoVSPt_resol", resolDir,
       "hFunctionResolCotgTheta_ResoVSPt_prof", functionResolDir,
       "resolCotgThetaVSpt", outputFile);
  // VS Eta
  draw("hResolCotgThetaGenVSMu_ResoVSEta_resol", resolDir,
       "hFunctionResolCotgTheta_ResoVSEta_prof", functionResolDir,
       "resolCotgThetaVSeta", outputFile);

  // sigmaPhi
  // --------
  resolDir = (TDirectory*) resolFile->Get("hResolPhiGenVSMu");
  functionResolDir = (TDirectory*) functionResolFile->Get("hFunctionResolPhi");
  // VS Pt
  draw("hResolPhiGenVSMu_ResoVSPt_resol", resolDir,
       "hFunctionResolPhi_ResoVSPt_prof", functionResolDir,
       "resolPhiVSpt", outputFile);
  // VS Eta
  draw("hResolPhiGenVSMu_ResoVSEta_resol", resolDir,
       "hFunctionResolPhi_ResoVSEta_prof", functionResolDir,
       "resolPhiVSeta", outputFile);
}
