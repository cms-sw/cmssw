#include "HistoFinder.cc"
#include "TLegend.h"
#include "THStack.h"
#include <vector>


/**
 * Base class used by Normalized and Merged. It sets up the canvas and the legend.
 */
class BaseNormalized {
 public:
  BaseNormalized() {}
  ~BaseNormalized() {}
  BaseNormalized( const TString & name, const TString & title ) :
    name_(name),
    title_(title),
    canvas_(new TCanvas(name+"_canvas",title+" canvas",1000,800)),
    legend_(new TLegend(0.779429,0.715556,0.979429,0.955556))
  {
    // legend_->SetHeader("");
    legend_->SetTextSize(0.04);
    // legend_->SetMarkerSize(1.4); // increase markersize so that they are visible
    legend_->SetFillColor(0); // Have a white background
  }
 protected:
  TString name_;
  TString title_;
  TCanvas * canvas_;  
  TLegend * legend_;
  std::vector<TH1*> histoList_;
  static Color_t colors_[4];
};

Color_t BaseNormalized::colors_[] = {kBlack, kRed, kBlue, kGreen};

/**
 * This class draws hitograms so that they are normalized and with different colors.
 * It also adds a legend.
 */
class Normalized : BaseNormalized {
 public:
  // Normalized() {}
  Normalized( const TString & name, const TString & title ) : BaseNormalized( name, title ) {}
  // virtual ~Normalized() {}
  void Add( TH1 * histo, const TString & name ) {
    histoList_.push_back(histo);
    legend_->AddEntry(histo, name);
  }
  // Do the loop here, so that we can use options like "errors"
  void Draw( const TString & xTitle = "", const TString & yTitle = "", const bool errors = false ) {

    // Create a new THStack so that it handle tha maximum
    // THStack stack(name_, title_);
    THStack * stack = new THStack(name_, title_);

    int colorIndex = 0;
    if( !(histoList_.empty()) ) {
      std::vector<TH1*>::iterator histoIter = histoList_.begin();
      for( ; histoIter != histoList_.end(); ++histoIter, ++colorIndex ) {
        TH1 * histo = *histoIter;
        if(errors) histo->Sumw2();
        // histo->SetNormFactor(1);
        if( colorIndex < 4 ) histo->SetLineColor(colors_[colorIndex]);
        else histo->SetLineColor(colorIndex);
        // Draw and get the maximum value
        TString normalizedHistoName(histo->GetName());
        TH1 * normalizedHisto = (TH1*)histo->Clone(normalizedHistoName+"clone");
        normalizedHisto->Scale(1/normalizedHisto->Integral());
        stack->Add(normalizedHisto);
      }
      // Take the maximum of all the drawed histograms
      // First we need to draw the histogram, or getAxis() will return 0... (see root code...)
      canvas_->Draw();
      canvas_->cd();
      stack->Draw("nostack");
      stack->GetYaxis()->SetTitleOffset(1.2);
      stack->GetYaxis()->SetTitle(yTitle);
      stack->GetXaxis()->SetTitle(xTitle);
      stack->GetXaxis()->SetTitleColor(kBlack);
      stack->Draw("nostack");
      legend_->Draw("same");

      canvas_->Update();
      canvas_->Draw();
      canvas_->ForceUpdate();
      //canvas_->Print("test.pdf");
      canvas_->Write();

    }
  }
};

/**
 * This class adds histograms into a merged one and draws it.
 * If the scaleFactors are not passed, the histograms are added
 * without rescaling. If the scaled factors are passed, the
 * histograms are scaled with those factors.
 */
class Merged : public BaseNormalized {
 public:
  Merged( const TString & name, const TString & title ) : BaseNormalized(name, title) {}
  // virtual ~Merged() {}
  void Add( TH1 * histo, const TString & name, const double & scaleFactorValue = 0 ) {
    histoList_.push_back(histo);
    legend_->AddEntry(histo, name);
    scaleFactor.push_back(scaleFactorValue);
  }
  void Draw( const TString & xTitle = "", const TString & yTitle = "", const bool errors = false ) {
    canvas_->cd();
    if( !(histoList_.empty()) ) {
      TString mergedName = histoList_[0]->GetName();
      mergedName+="_Merged";
      TH1 * histo_Merged = (TH1*)histoList_[0]->Clone(mergedName);
      histo_Merged->Reset();
      std::vector<TH1*>::iterator histoIter = histoList_.begin();
      int scaleFactorIndex = 0;
      for( ; histoIter != histoList_.end(); ++histoIter, ++scaleFactorIndex ) {
        TH1 * histo = *histoIter;
        if( scaleFactor[scaleFactorIndex] != 0 ) histo_Merged->Add(histo, scaleFactor[scaleFactorIndex]/histo->Integral());
        else histo_Merged->Add(histo);
      }
      if(errors) histo_Merged->Sumw2();
      histo_Merged->GetXaxis()->SetTitle(xTitle);
      histo_Merged->GetYaxis()->SetTitleOffset(1.2);
      histo_Merged->GetYaxis()->SetTitle(yTitle);
      histo_Merged->Draw();
      canvas_->Draw();
      canvas_->Write();
    }
  }
 protected:
  std::vector<double> scaleFactor;
};

/**
 * Small function used to draw J/Psi, Y and Z histograms, both superimposed and merged. Runs on ResolutionAnalyzer files.
 */
void drawKinematics( TFile ** inputFileList, int * inputScaleList, const TString & name, const TString & title, const TString & xTitle = "", const TString & yTitle = "", const TString & yTitleMerged = "", const TString & legPreText = "" ) {
  std::cout << "Drawing: " << name << std::endl;
  HistoFinder finder;

  TH1F * histoPt_JPsi = (TH1F*)finder(name, inputFileList[0]);
  TH1F * histoPt_Y = (TH1F*)finder(name, inputFileList[1]);
  TH1F * histoPt_Z = (TH1F*)finder(name, inputFileList[2]);

  Normalized resonancePt( name, title );
  resonancePt.Add(histoPt_JPsi, legPreText+"J/#Psi");
  resonancePt.Add(histoPt_Y, legPreText+"Y");
  resonancePt.Add(histoPt_Z, legPreText+"Z");
  resonancePt.Draw(xTitle, yTitle);

  if( inputScaleList != 0 ) {
    Merged resonancePtMerged( name+"Merged", title+" merged" );
    resonancePtMerged.Add(histoPt_JPsi, "J/#Psi", inputScaleList[0]);
    resonancePtMerged.Add(histoPt_Y, "Y", inputScaleList[1]);
    resonancePtMerged.Add(histoPt_Z, "Z", inputScaleList[2]);
    resonancePtMerged.Draw(xTitle, yTitleMerged);
  }
}

/**
 * This function draws histograms of kinematic characteristics of the Z, Y and J/Psi samples
 */
void KinematicsComparison() {
  gROOT->SetBatch(true);
  gROOT->SetStyle("Plain");

  TFile * inputFileList[3] = {
    new TFile("ResolutionAnalyzer_JPsi.root","READ"),
    new TFile("ResolutionAnalyzer_Y.root","READ"),
    new TFile("ResolutionAnalyzer_Z.root","READ")
  };

  TFile * outputFile = new TFile("KinematicsComparison.root", "RECREATE");

  int inputScaleList[3] = { 126629, 48122, 1667 };

  outputFile->cd();

  // Resonance comparison histograms
  // -------------------------------
  TDirectory * resonancesDir = outputFile->mkdir("Resonances");
  resonancesDir->cd();
  // drawKinematics( inputFileList, 0, "RecoResonance_Pt", "resonance Pt", "Pt(GeV)", "arbitrary units", "expected number of events in 3.3/pb" );
  // Pt
  drawKinematics( inputFileList, inputScaleList, "RecoResonance_Pt", "resonance Pt", "pt(GeV)", "arbitrary units", "expected number of events in 3.3/pb" );
  // Eta
  drawKinematics( inputFileList, inputScaleList, "RecoResonance_Eta", "resonance #eta", "#eta", "arbitrary units", "expected number of events in 3.3/pb" );
  // Phi
  drawKinematics( inputFileList, inputScaleList, "RecoResonance_Phi", "resonance #phi", "#phi", "arbitrary units", "expected number of events in 3.3/pb" );

  // Histograms of muons from the resonances
  // ---------------------------------------
  TDirectory * resonancesMuonsDir = outputFile->mkdir("ResonanceMuons");
  resonancesMuonsDir->cd();
  // DeltaPt
  drawKinematics( inputFileList, 0, "RecoResonanceMuons_Pt", "pt of muons from resonance", "pt(GeV)", "arbitrary units", "", "muons from " );
  // Eta
  drawKinematics( inputFileList, 0, "RecoResonanceMuons_Eta", "#eta of muons from resonance", "#eta", "arbitrary units", "", "muons from " );
  // Phi
  drawKinematics( inputFileList, 0, "RecoResonanceMuons_Phi", "#phi of muons from resonance", "#phi", "arbitrary units", "", "muons from " );

  // Histograms of Deltas of muons from the resonances
  // -------------------------------------------------
  TDirectory * resonancesMuonsDeltasDir = outputFile->mkdir("ResonanceMuonsDeltas");
  resonancesMuonsDeltasDir->cd();
  // DeltaCotgTheta
  drawKinematics( inputFileList, 0, "DeltaRecoResonanceMuons_DeltaCotgTheta", "#Delta Cotg(#theta) of muons from resonance", "Cotg(#theta)", "arbitrary units", "", "muons from " );
  // DeltaTheta
  drawKinematics( inputFileList, 0, "DeltaRecoResonanceMuons_DeltaTheta", "#Delta#theta of muons from resonance", "#Delta#theta", "arbitrary units", "", "muons from " );
  // DeltaEta
  drawKinematics( inputFileList, 0, "DeltaRecoResonanceMuons_DeltaEta", "|#Delta#eta| of muons from resonance", "|#Delta#eta|", "arbitrary units", "", "muons from " );
  // DeltaEtaSign
  drawKinematics( inputFileList, 0, "DeltaRecoResonanceMuons_DeltaEtaSign", "#Delta#eta with sign of muons from resonance", "#Delta#eta", "arbitrary units", "", "muons from " );
  // DeltaPhi
  drawKinematics( inputFileList, 0, "DeltaRecoResonanceMuons_DeltaPhi", "#Delta#phi of muons from resonance", "#Delta#phi", "arbitrary units", "", "muons from " );
  // DeltaR
  drawKinematics( inputFileList, 0, "DeltaRecoResonanceMuons_DeltaR", "#Delta R of muons from resonance", "#Delta R", "arbitrary units", "", "muons from " );
}
