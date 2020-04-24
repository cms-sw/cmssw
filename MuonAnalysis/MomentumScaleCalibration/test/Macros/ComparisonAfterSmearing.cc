// #include "TFile.h"
// #include "TDirectory.h"
// #include "TCanvas.h"
// #include "TH1D.h"
// #include "TProfile.h"

/**
 * This uncompiled macro draws the resolution histograms superimposed on the same canvas for the various quantities. <br>
 * It draws the reco-gen derived resolution (true resolution), those from the resolution functions from 0_MuScleFit.root
 * and those from the resolution functions from 1_MuScleFit.root for a comparison of the change aligned and after the smearing.
 */

draw( const TString & resolName, TDirectory * resolDir, TDirectory * resolDirMisaligned,
      const TString & functionResolName, TDirectory * functionResolDir, TDirectory * functionResolDirMisaligned,
      const TString & canvasNameMC, const TString & canvasNameFunc, TFile * outputFile,
      const TString & title = "", const TString & xAxisTitle = "", const TString & yAxisTitle = "", double Ymax,
      TDirectory * functionResolDirSmeared = 0, TDirectory * resolDirSmeared = 0, double Xmin = 0, double Xmax = 80)
{
  TProfile * functionResolVSpt = (TProfile*) functionResolDir->Get(functionResolName);
  TProfile * functionResolVSptMisaligned = (TProfile*) functionResolDirMisaligned->Get(functionResolName);

  TProfile * functionResolVsptSmeared = 0;
  if( functionResolDirSmeared != 0 ) functionResolVSptSmeared = (TProfile*) functionResolDirSmeared->Get(functionResolName);
  TH1D * resolVSpt = (TH1D*) resolDir->Get(resolName);
  TH1D * resolVSptMisaligned = (TH1D*) resolDirMisaligned->Get(resolName);
  TH1D * resolVSptSmeared = 0;
  if( resolDirSmeared != 0 ) {
    resolVSptSmeared = (TH1D*) resolDirSmeared->Get(resolName);
  }

  TString resolVSptName("from reco-gen comparison");
  TCanvas * c = new TCanvas(canvasNameMC, canvasNameMC, 1000, 800);
  c->cd();
  TLegend * legend = new TLegend(0.55,0.71,0.98,1.);
  legend->SetTextSize(0.03);
  legend->SetFillColor(0); // Have a white background
  if( resolDirSmeared == 0 ) legend->AddEntry(resolVSpt, resolVSptName);
  else legend->AddEntry(resolVSpt, resolVSptName+" ideal");
  resolVSpt->SetTitle(title);
  resolVSpt->Draw();
  resolVSpt->GetXaxis()->SetTitle(xAxisTitle);
  resolVSpt->GetYaxis()->SetTitle(yAxisTitle);
  resolVSpt->GetXaxis()->SetRangeUser(Xmin,Xmax);
  resolVSpt->SetMaximum(Ymax);
  resolVSpt->SetMinimum(0);
  resolVSpt->Draw();
  resolVSptMisaligned->SetLineColor(kRed);
  legend->AddEntry(resolVSptMisaligned, resolVSptName + " fake data");
  resolVSptMisaligned->Draw("SAME");
  if( resolDirSmeared != 0 ) {
    resolVSptSmeared->SetLineColor(kBlue);
    legend->AddEntry(resolVSptSmeared, resolVSptName + " smeared");
    resolVSptSmeared->Draw("SAME");
  }
  legend->Draw();
  outputFile->cd();
  c->Write();

  TCanvas * c2 = new TCanvas(canvasNameFunc, canvasNameFunc, 1000, 800);
  c2->cd();
  TLegend * legend2 = new TLegend(0.55,0.71,0.98,1.);
  legend2->SetTextSize(0.03);
  legend2->SetFillColor(0); // Have a white background
  TString functionLegendName("from resolution function");
  TString functionLegendNameIdeal;
  TString functionLegendNameSmeared;
  TString functionLegendNameMisaligned;

  functionLegendNameIdeal = functionLegendName + " ideal";
  legend2->AddEntry(functionResolVSpt, functionLegendNameIdeal);
  functionResolVSpt->GetXaxis()->SetTitle(xAxisTitle);
  functionResolVSpt->GetYaxis()->SetTitle(yAxisTitle);  
  functionResolVSpt->GetXaxis()->SetRangeUser(Xmin,Xmax);
  functionResolVSpt->SetMaximum(Ymax);
  functionResolVSpt->SetMinimum(0);
  functionResolVSpt->Draw();

  functionLegendNameMisaligned = functionLegendName + " fake data";
  functionResolVSptMisaligned->SetMarkerColor(kRed);
  functionResolVSptMisaligned->SetLineColor(kRed);
  legend2->AddEntry(functionResolVSptMisaligned, functionLegendNameMisaligned);
  functionResolVSptMisaligned->Draw("same");

  functionLegendNameSmeared = functionLegendName + " smeared";
  functionResolVSptSmeared->SetMarkerColor(kBlue);
  functionResolVSptSmeared->SetLineColor(kBlue);
  legend2->AddEntry(functionResolVSptSmeared, functionLegendNameSmeared);  
  functionResolVSptSmeared->Draw("same");
  
  legend2->Draw();
  // c->Draw();
  outputFile->cd();
  c2->Write();
}

TF1 * expRelativisticBWintPhotFit(const std::string & index)
{
  ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator * fobj = new ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator;
  TF1 * functionToFit = new TF1(("functionToFit"+index).c_str(), fobj, 60, 120, fobj->parNum(), "ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator");
  functionToFit->SetParameter(0, 2.);
  functionToFit->SetParameter(1, 90.);
  functionToFit->SetParameter(2, 800.);
  functionToFit->SetParameter(3, 1.);
  functionToFit->SetParameter(4, 0.);
  functionToFit->SetParameter(5, 0.);

  functionToFit->SetParLimits(3, 0., 1.);
  functionToFit->SetParLimits(4, 0., 1.);

  return functionToFit;
}

// Product between an exponential term and the relativistic Breit-Wigner with Z/gamma interference term and photon propagator
// --------------------------------------------------------------------------------------------------------------------------
class ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator
{
 public:
  ExpRelativisticBWwithZGammaInterferenceAndPhotonPropagator()
  {
    parNum_ = 6;
    twoOverPi_ = 2./TMath::Pi();
  }
  double operator() (double *x, double *p)
  {

    // if( p[3]+p[4] > 1 ) return -10000.;

    double squaredMassDiff = pow((x[0]*x[0] - p[1]*p[1]), 2);
    double denominator = squaredMassDiff + pow(x[0], 4)*pow(p[0]/p[1], 2);
    return p[2]*exp(-p[5]*x[0])*( p[3]*twoOverPi_*pow(p[1]*p[0], 2)/denominator + (1-p[3]-p[4])*p[1]*squaredMassDiff/denominator + p[4]/(x[0]*x[0]));
  }
  int parNum() const { return parNum_; }
 protected:
  int parNum_;
  double twoOverPi_;
};

drawZ( const TString & HistoName, TFile * AlignedFile, TFile * MisalignedFile,
       const TString & canvasName, TFile * outputFile, const TString & stringName,
       const TString & title = "", const TString & xAxisTitle = "", const TString & yAxisTitle = "", double Xmin = 0, double Xmax = 120, int rebinX = 1,
	  TFile * SmearedFile = 0 )
{
  TH1F * Aligned = (TH1F*) AlignedFile->Get(HistoName);
  TH1F * Misaligned = (TH1F*) MisalignedFile->Get(HistoName);
  
  TH1F * Smeared = 0;
  if( SmearedFile != 0 ) Smeared = (TH1F*) SmearedFile->Get(HistoName);
  
  TCanvas * c = new TCanvas(canvasName, canvasName, 1000, 800);
  c->Divide(1,2);
  c->cd(1);
  TLegend * legend3 = new TLegend(0.65,0.55,0.98,1.);
  legend3->SetTextSize(0.06);
  legend3->SetFillColor(0); // Have a white background
  TString LegendName(stringName);
  TString LegendNameAligned;
  TString LegendNameSmeared;
  TString LegendNameMisaligned;
  
  LegendNameAligned = LegendName + "ideal MC";
  legend3->AddEntry(Aligned, LegendNameAligned);
  Aligned->Rebin(rebinX);
  Aligned->GetXaxis()->SetRangeUser(Xmin,Xmax);
  //  Aligned->Sumw2();
  Aligned->GetXaxis()->SetTitle(xAxisTitle);
  Aligned->GetYaxis()->SetTitle(yAxisTitle);  
  Aligned->Draw("e");
  if(HistoName == "hRecBestRes_Mass"){
    TF1 * functionToFitId = expRelativisticBWintPhotFit("Ideal");
    Aligned->Fit(functionToFitId, "MN", "", 60, 120);  
    functionToFitId->Draw("same");
  }

  LegendNameMisaligned = LegendName + "fake data";
  Misaligned->SetMarkerColor(kRed);
  Misaligned->SetLineColor(kRed);
  legend3->AddEntry(Misaligned, LegendNameMisaligned);
  Misaligned->Rebin(rebinX);
  //  Misaligned->Sumw2();
  Misaligned->Draw("esames");
  if(HistoName == "hRecBestRes_Mass"){
    TF1 * functionToFitFake = expRelativisticBWintPhotFit("Fake");
    Misaligned->Fit(functionToFitFake, "MN", "", 60, 120);  
    functionToFitFake->SetLineColor(kRed);
    functionToFitFake->Draw("same");
  }

  LegendNameSmeared = LegendName + "smeared MC";
  Smeared->SetMarkerColor(kBlue);
  Smeared->SetLineColor(kBlue);
  legend3->AddEntry(Smeared, LegendNameSmeared);  
  Smeared->Rebin(rebinX);
  //  Smeared->Sumw2();
  Smeared->Draw("esames");
  if(HistoName == "hRecBestRes_Mass"){
    TF1 * functionToFitSmear = expRelativisticBWintPhotFit("Smear");
    Smeared->Fit(functionToFitSmear, "MN", "", 60, 120);  
    functionToFitSmear->SetLineColor(kBlue);
    functionToFitSmear->Draw("same");
  }

  legend3->Draw();

  c->cd(2);
  TH1F *RatioIdFake = (TH1F*) Aligned->Clone();
  TH1F *RatioSmeFake = (TH1F*) Aligned->Clone();

  TString LegendNameRatio1, LegendNameRatio2;
  TLegend * legend4 = new TLegend(0.65,0.71,0.98,1.);
  LegendNameRatio1 = "Z pt MC/Z pt fake";
  RatioIdFake->Add(Aligned,Misaligned,1,-1);  
  //  RatioIdFake->Sumw2();
  //  RatioIdFake->Divide(Aligned,Misaligned);
  
  RatioIdFake->SetLineColor(kRed);
  RatioIdFake->Draw("");  
  legend4->AddEntry(RatioIdFake,LegendNameRatio1); 
  LegendNameRatio2 = "Z pt MC smeared/Z pt fake";
  RatioSmeFake->Add(Smeared,Misaligned,1,-1);  
  // RatioSmeFake->Divide(Smeared,Misaligned);
  RatioSmeFake->SetLineColor(kBlue);
  RatioSmeFake->SetTitle("diff");
  RatioSmeFake->Draw("sames");  
  legend4->AddEntry(RatioSmeFake,LegendNameRatio2); 
  legend4->Draw();

  outputFile->cd();
  c->Write();

}

/**
 * This macro compares the resolutions found with the resolution functions with those
 * obtained from recMu-genMu comparison in MuScleFit.
 * The true resolutions are writted in the "redrawed.root" file by the ResolDraw.cc macro.
 */
void ComparisonAfterSmearing(const TString & stringNumAligned = "Ideal", const TString & stringNumMisaligned = "Fake", const TString & stringNumSmeared = "Smear") {

  // Remove the stat box
  gStyle->SetOptStat(0);

  TFile * outputFile = new TFile("ComparisonAfterSmearing.root", "RECREATE");

  TFile * resolFileAligned = new TFile("redrawed_"+stringNumAligned+".root", "READ");
  TFile * resolFileMisaligned = new TFile("redrawed_"+stringNumMisaligned+".root", "READ");
  TFile * resolFileSmeared = 0;
  resolFileSmeared = new TFile("redrawed_"+stringNumSmeared+".root", "READ");
  TFile * functionResolFileAligned = new TFile(stringNumAligned+"_MuScleFit.root", "READ");
  TFile * functionResolFileMisaligned = new TFile(stringNumMisaligned+"_MuScleFit.root", "READ");
  TFile * functionResolFileSmeared = new TFile(stringNumSmeared+"_MuScleFit.root", "READ");

  TDirectory * resolDirAligned = 0;
  TDirectory * resolDirMisaligned = 0;
  TDirectory * resolDirSmeared = 0;
  TDirectory * functionResolDirAligned = 0;
  TDirectory * functionResolDirMisaligned = 0;
  TDirectory * functionResolDirSmeared = 0;

  //Mass lineshape and pt

  drawZ("hRecBestRes_Mass", functionResolFileAligned, functionResolFileMisaligned ,
	"ZMass", outputFile, "Z Mass from ",
	"Di-muon Mass", "Di-muon Mass [GeV]", "a.u.", 60, 120, 20,
	functionResolFileSmeared);

  drawZ("hRecBestRes_Pt", functionResolFileAligned, functionResolFileMisaligned ,
	"ZPt", outputFile, "Z pt from ",
	"Z Transverse momentum", "Z pt [GeV]", "a.u.", 1, 100, 1,
	functionResolFileSmeared);

  // SigmaMass/Mass
  // --------------
  resolDirAligned = (TDirectory*) resolFileAligned->Get("DeltaMassOverGenMassVsPt");
  resolDirMisaligned = (TDirectory*) resolFileMisaligned->Get("DeltaMassOverGenMassVsPt");
  if( resolFileSmeared == 0 ) resolDirSmeared = 0;
  else resolDirSmeared = (TDirectory*) resolFileSmeared->Get("DeltaMassOverGenMassVsPt");
  functionResolDirAligned = (TDirectory*) functionResolFileAligned->Get("hFunctionResolMassVSMu");
  functionResolDirMisaligned = (TDirectory*) functionResolFileMisaligned->Get("hFunctionResolMassVSMu");
  functionResolDirSmeared = (TDirectory*) functionResolFileSmeared->Get("hFunctionResolMassVSMu");
  // Vs Pt
  draw("DeltaMassOverGenMassVsPt_resol", resolDirAligned, resolDirMisaligned,
       "hFunctionResolMassVSMu_ResoVSPt_prof", functionResolDirAligned, functionResolDirMisaligned,
       "massResolVSptMC","massResolVSptFunc", outputFile,
       "resolution on mass vs pt",
       "muon pt(GeV)", "res #sigmaM/M",0.09,
       functionResolDirSmeared, resolDirSmeared );
  // VsEta
  resolDirAligned = (TDirectory*) resolFileAligned->Get("DeltaMassOverGenMassVsEta");
  resolDirMisaligned = (TDirectory*) resolFileMisaligned->Get("DeltaMassOverGenMassVsEta");

  if( resolFileSmeared == 0 ) resolDirSmeared = 0;
  else resolDirSmeared = (TDirectory*) resolFileSmeared->Get("DeltaMassOverGenMassVsEta");
  draw("DeltaMassOverGenMassVsEta_resol", resolDirAligned, resolDirMisaligned,
       "hFunctionResolMassVSMu_ResoVSEta_prof", functionResolDirAligned, functionResolDirMisaligned,
       "massResolVSetaMC", "massResolVSetaFunc", outputFile,
       "resolution on mass vs #eta",
       "muon #eta", "#sigmaM/M",0.09,
       functionResolDirSmeared, resolDirSmeared, -3, 3 );


  // sigmaPt/Pt
  // ----------
  resolDirAligned = (TDirectory*) resolFileAligned->Get("hResolPtGenVSMu");
  resolDirMisaligned = (TDirectory*) resolFileMisaligned->Get("hResolPtGenVSMu");

  if( resolFileSmeared == 0 ) resolDirSmeared = 0;
  else resolDirSmeared = (TDirectory*) resolFileSmeared->Get("hResolPtGenVSMu");
  functionResolDirAligned = (TDirectory*) functionResolFileAligned->Get("hFunctionResolPt");
  functionResolDirMisaligned = (TDirectory*) functionResolFileMisaligned->Get("hFunctionResolPt");
  functionResolDirSmeared = (TDirectory*) functionResolFileSmeared->Get("hFunctionResolPt");

  //=====> All
  // VS Pt
  draw("hResolPtGenVSMu_ResoVSPt_resol", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSPt_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSptMC", "resolPtVSptFunc", outputFile,
       "resolution on pt vs pt",
       "muon pt(GeV)", "#sigmapt/pt",0.09,
       functionResolDirSmeared, resolDirSmeared );
  // VS Pt RMS
  draw("hResolPtGenVSMu_ResoVSPt_resolRMS", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSPt_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSptRMSMC", "resolPtVSptRMSFunc", outputFile,
       "resolution on pt vs pt",
       "muon pt(GeV)", "#sigmapt/pt",0.09,
       functionResolDirSmeared, resolDirSmeared );

  //=====> Barrel
  // VS Pt
  draw("hResolPtGenVSMu_ResoVSPt_Bar_resol", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSPt_Bar_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSptBarMC", "resolPtVSptBarFunc", outputFile,
       "resolution on pt vs pt, barrel",
       "muon pt(GeV)", "#sigmapt/pt",0.09,
       functionResolDirSmeared, resolDirSmeared );
  // VS Pt RMS
  draw("hResolPtGenVSMu_ResoVSPt_Bar_resolRMS", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSPt_Bar_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSptBarRMSMC", "resolPtVSptBarRMSFunc", outputFile,
       "resolution on pt vs pt, barrel",
       "muon pt(GeV)", "#sigmapt/pt",0.09,
       functionResolDirSmeared, resolDirSmeared );

  
//   //=====> Endcap 1
//   // VS Pt
//   draw("hResolPtGenVSMu_ResoVSPt_Endc_1.7_resol", resolDirAligned, resolDirMisaligned,
//        "hFunctionResolPt_ResoVSPt_Endc_1.7_prof", functionResolDirAligned,functionResolDirMisaligned,
//        "resolPtVSptEndc_1.7", outputFile,
//        "resolution on pt vs pt, endcap",
//        "muon pt(GeV)", "#sigmapt/pt",0.09,
//        functionResolDirSmeared, resolDirSmeared );
//   // VS Pt RMS
//   draw("hResolPtGenVSMu_ResoVSPt_Endc_1.7_resolRMS", resolDirAligned, resolDirMisaligned,
//        "hFunctionResolPt_ResoVSPt_Endc_1.7_prof", functionResolDirAligned, functionResolDirMisaligned,
//        "resolPtVSptEndc1.7RMS", outputFile,
//        "resolution on pt vs pt, endcap",
//        "muon pt(GeV)", "#sigmapt/pt",0.09,
//        functionResolDirSmeared, resolDirSmeared );

//   //=====> Endcap 2
//   // VS Pt
//   draw("hResolPtGenVSMu_ResoVSPt_Endc_2.0_resol", resolDirAligned, resolDirMisaligned,
//        "hFunctionResolPt_ResoVSPt_Endc_2.0_prof", functionResolDirAligned, functionResolDirMisaligned,
//        "resolPtVSptEndc2.0", outputFile,
//        "resolution on pt vs pt, endcap",
//        "muon pt(GeV)", "#sigmapt/pt",0.09,
//        functionResolDirSmeared, resolDirSmeared );
//   // VS Pt RMS
//   draw("hResolPtGenVSMu_ResoVSPt_Endc_2.0_resolRMS", resolDirAligned, resolDirMisaligned,
//        "hFunctionResolPt_ResoVSPt_Endc_2.0_prof", functionResolDirAligned, functionResolDirMisaligned,
//        "resolPtVSptEndc2.0RMS", outputFile,
//        "resolution on pt vs pt, endcap",
//        "muon pt(GeV)", "#sigmapt/pt",0.09,
//        functionResolDirSmeared, resolDirSmeared );

//   //=====> Endcap 3
//   // VS Pt
//   draw("hResolPtGenVSMu_ResoVSPt_Endc_2.4_resol", resolDirAligned, resolDirMisaligned,
//        "hFunctionResolPt_ResoVSPt_Endc_2.4_prof", functionResolDirAligned, functionResolDirMisaligned,
//        "resolPtVSptEndc2.4", outputFile,
//        "resolution on pt vs pt, endcap",
//        "muon pt(GeV)", "#sigmapt/pt",0.09,
//        functionResolDirSmeared, resolDirSmeared );
//   // VS Pt RMS
//   draw("hResolPtGenVSMu_ResoVSPt_Endc_2.4_resolRMS", resolDirAligned,resolDirMisaligned, 
//        "hFunctionResolPt_ResoVSPt_Endc_2.4_prof", functionResolDirAligned, functionResolDirMisaligned,
//        "resolPtVSptEndc2.4RMS", outputFile,
//        "resolution on pt vs pt, endcap",
//        "muon pt(GeV)", "#sigmapt/pt",0.09,
//        functionResolDirSmeared, resolDirSmeared );


  //=====> overlap
  // VS Pt
  draw("hResolPtGenVSMu_ResoVSPt_Ovlap_resol", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSPt_Ovlap_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSptOvlapMC",  "resolPtVSptOvlapFunc", outputFile,
       "resolution on pt vs pt, overlap",
       "muon pt(GeV)", "#sigmapt/pt",0.09,
       functionResolDirSmeared, resolDirSmeared );
  // VS Pt RMS
  draw("hResolPtGenVSMu_ResoVSPt_Ovlap_resolRMS", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSPt_Ovlap_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSptOvlapRMSMC", "resolPtVSptOvlapRMSFunc", outputFile,
       "resolution on pt vs pt",
       "muon pt(GeV)", "#sigmapt/pt",0.09,
       functionResolDirSmeared, resolDirSmeared );

  // VS Eta
  draw("hResolPtGenVSMu_ResoVSEta_resol", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSEta_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSetaMC", "resolPtVSetaFunc", outputFile,
       "resolution on pt vs #eta",
       "muon #eta", "#sigmapt/pt",0.15,
       functionResolDirSmeared, resolDirSmeared, -3, 3 );
  // VS Eta RMS
  draw("hResolPtGenVSMu_ResoVSEta_resolRMS", resolDirAligned, resolDirMisaligned,
       "hFunctionResolPt_ResoVSEta_prof", functionResolDirAligned, functionResolDirMisaligned,
       "resolPtVSetaRMSMC", "resolPtVSetaRMSFunc", outputFile,
       "resolution on pt vs #eta",
       "muon #eta", "#sigmapt/pt",0.15,
       functionResolDirSmeared, resolDirSmeared, -3, 3  );


}
