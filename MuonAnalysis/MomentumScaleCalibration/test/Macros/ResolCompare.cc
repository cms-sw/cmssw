// #include "TFile.h"
// #include "TDirectory.h"
// #include "TCanvas.h"
// #include "TH1D.h"
// #include "TProfile.h"

/**
 * This uncompiled macro draws the resolution histograms superimposed on the same canvas for the various quantities. <br>
 * It draws the reco-gen derived resolution (true resolution), those from the resolution functions from 0_MuScleFit.root
 * and those from the resolution functions from 1_MuScleFit.root for a comparison of the change before and after the fit.
 */
draw( const TString & resolName, TDirectory * resolDir,
      const TString & functionResolName, TDirectory * functionResolDir,
      const TString & canvasName, TFile * outputFile,
      const TString & title = "", const TString & xAxisTitle = "", const TString & yAxisTitle = "", double Ymax,
      TDirectory * functionResolDirAfter = 0 )
{
  TH1D * resolVSpt = (TH1D*) resolDir->Get(resolName);
  TProfile * functionResolVSpt = (TProfile*) functionResolDir->Get(functionResolName);
  TProfile * functionResolVsptAfter = 0;
  if( functionResolDirAfter != 0 ) functionResolVSptAfter = (TProfile*) functionResolDirAfter->Get(functionResolName);

  TCanvas * c = new TCanvas(canvasName, canvasName, 1000, 800);
  c->cd();
  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background
  legend->AddEntry(resolVSpt, "from reco-gen comparison");
  resolVSpt->SetTitle(title);
  resolVSpt->GetXaxis()->SetTitle(xAxisTitle);
  resolVSpt->GetYaxis()->SetTitle(yAxisTitle);
  resolVSpt->SetMaximum(Ymax);
  resolVSpt->SetMinimum(-Ymax/2);
  resolVSpt->Draw();

  TString functionLegendName("from resolution function");
  TString functionLegendNameAfter;
  functionResolVSpt->SetMarkerColor(kRed);
  functionResolVSpt->SetLineColor(kRed);
  if( functionResolDirAfter != 0 ) {
    functionLegendNameAfter = functionLegendName + " after";
    functionLegendName += " before";
    functionResolVSptAfter->SetMarkerColor(kBlue);
    functionResolVSptAfter->SetLineColor(kBlue);
  }
  legend->AddEntry(functionResolVSpt, functionLegendName);
  functionResolVSpt->Draw("same");

  if( functionResolDirAfter != 0 ) {
    legend->AddEntry(functionResolVSptAfter, functionLegendNameAfter);
    functionResolVSptAfter->Draw("same");
  }

  legend->Draw("same");
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

  // Remove the stat box
  gStyle->SetOptStat(0);

  TFile * outputFile = new TFile("ComparedResol.root", "RECREATE");

  TFile * resolFile = new TFile("redrawed.root", "READ");
  TFile * functionResolFileBefore = new TFile("0_MuScleFit.root", "READ");
  TFile * functionResolFileAfter = new TFile("1_MuScleFit.root", "READ");

  TDirectory * resolDir = 0;
  TDirectory * functionResolDirBefore = 0;
  TDirectory * functionResolDirAfter = 0;
  // sigmaPt
  // -------
  resolDir = (TDirectory*) resolFile->Get("hResolPtGenVSMu");
  functionResolDirBefore = (TDirectory*) functionResolFileBefore->Get("hFunctionResolPt");
  functionResolDirAfter = (TDirectory*) functionResolFileAfter->Get("hFunctionResolPt");
  // VS Pt
  draw("hResolPtGenVSMu_ResoVSPt_resol", resolDir,
       "hFunctionResolPt_ResoVSPt_prof", functionResolDirBefore,
       "resolPtVSpt", outputFile,
       "resolution on pt vs pt",
       "pt(GeV)", "#sigmapt",0.05,
       functionResolDirAfter );
  // VS Eta
  draw("hResolPtGenVSMu_ResoVSEta_resol", resolDir,
       "hFunctionResolPt_ResoVSEta_prof", functionResolDirBefore,
       "resolPtVSeta", outputFile,
       "resolution on pt vs #eta",
       "#eta", "#sigmapt",0.05,
       functionResolDirAfter );

  // sigmaCotgTheta
  // --------------
  resolDir = (TDirectory*) resolFile->Get("hResolCotgThetaGenVSMu");
  functionResolDirBefore = (TDirectory*) functionResolFileBefore->Get("hFunctionResolCotgTheta");
  functionResolDirAfter = (TDirectory*) functionResolFileAfter->Get("hFunctionResolCotgTheta");
  // VS Pt
  draw("hResolCotgThetaGenVSMu_ResoVSPt_resol", resolDir,
       "hFunctionResolCotgTheta_ResoVSPt_prof", functionResolDirBefore,
       "resolCotgThetaVSpt", outputFile,
       "resolution on cotg(#theta) vs pt",
       "pt(GeV)", "#sigmacotg(#theta)",0.01,
       functionResolDirAfter );
  // VS Eta
  draw("hResolCotgThetaGenVSMu_ResoVSEta_resol", resolDir,
       "hFunctionResolCotgTheta_ResoVSEta_prof", functionResolDirBefore,
       "resolCotgThetaVSeta", outputFile,
       "resolution on cotg(#theta) vs #eta",
       "#eta", "#sigmacotg(#theta)",0.01,
       functionResolDirAfter );
  // sigmaPhi
  // --------
  resolDir = (TDirectory*) resolFile->Get("hResolPhiGenVSMu");
  functionResolDirBefore = (TDirectory*) functionResolFileBefore->Get("hFunctionResolPhi");
  functionResolDirAfter = (TDirectory*) functionResolFileAfter->Get("hFunctionResolPhi");
  // VS Pt
  draw("hResolPhiGenVSMu_ResoVSPt_resol", resolDir,
       "hFunctionResolPhi_ResoVSPt_prof", functionResolDirBefore,
       "resolPhiVSpt", outputFile,
       "resolution on #phi vs pt",
       "pt(GeV)", "#sigma#phi",0.01,
       functionResolDirAfter );
  // VS Eta
  draw("hResolPhiGenVSMu_ResoVSEta_resol", resolDir,
       "hFunctionResolPhi_ResoVSEta_prof", functionResolDirBefore,
       "resolPhiVSeta", outputFile,
       "resolution on #phi vs #eta",
       "#eta", "#sigma#phi",0.01,
       functionResolDirAfter );
}
