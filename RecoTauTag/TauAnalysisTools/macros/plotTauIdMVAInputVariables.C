
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TH1.h>
#include <TH2.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TPaveText.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TMath.h>
#include <TROOT.h>
#include <TSystem.h>

#include "TMVA/Factory.h"
#include "TMVA/Reader.h"
#include "TMVA/Tools.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include <limits>

//-------------------------------------------------------------------------------
void normalizeHistogram(TH1* histogram)
{
  if ( histogram->Integral() > 0. ) {
    if ( !histogram->GetSumw2N() ) histogram->Sumw2();
    histogram->Scale(1./histogram->Integral());
  } else {
    std::cout << "integral(" << histogram->GetName() << ") = " << histogram->Integral() << " !!" << std::endl;
    assert(0);
  }
}

TH1* compRatioHistogram(const std::string& ratioHistogramName, const TH1* numerator, const TH1* denominator)
{
  assert(numerator->GetDimension() == denominator->GetDimension());
  assert(numerator->GetNbinsX() == denominator->GetNbinsX());

  TH1* histogramRatio = (TH1*)numerator->Clone(ratioHistogramName.data());
  histogramRatio->Divide(denominator);

  int nBins = histogramRatio->GetNbinsX();
  for ( int iBin = 1; iBin <= nBins; ++iBin ){
    double binContent = histogramRatio->GetBinContent(iBin);
    histogramRatio->SetBinContent(iBin, binContent - 1.);
  }

  histogramRatio->SetLineColor(numerator->GetLineColor());
  histogramRatio->SetLineWidth(numerator->GetLineWidth());
  histogramRatio->SetMarkerColor(numerator->GetMarkerColor());
  histogramRatio->SetMarkerStyle(numerator->GetMarkerStyle());

  return histogramRatio;
}

void showDistribution(double canvasSizeX, double canvasSizeY,
		      TH1* histogram_ref, const std::string& legendEntry_ref,
		      TH1* histogram2, const std::string& legendEntry2,
		      TH1* histogram3, const std::string& legendEntry3,
		      TH1* histogram4, const std::string& legendEntry4,
		      TH1* histogram5, const std::string& legendEntry5,
		      TH1* histogram6, const std::string& legendEntry6,
		      bool useLogScaleX, const std::string& xAxisTitle, double xAxisOffset,
		      bool useLogScaleY, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		      double legendX0, double legendY0, 
		      const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.12);
  canvas->SetBottomMargin(0.12);

  TPad* topPad = new TPad("topPad", "topPad", 0.00, 0.35, 1.00, 1.00);
  topPad->SetFillColor(10);
  topPad->SetTopMargin(0.04);
  topPad->SetLeftMargin(0.15);
  topPad->SetBottomMargin(0.03);
  topPad->SetRightMargin(0.05);
  topPad->SetLogx(useLogScaleX);
  topPad->SetLogy(useLogScaleY);

  TPad* bottomPad = new TPad("bottomPad", "bottomPad", 0.00, 0.00, 1.00, 0.35);
  bottomPad->SetFillColor(10);
  bottomPad->SetTopMargin(0.02);
  bottomPad->SetLeftMargin(0.15);
  bottomPad->SetBottomMargin(0.24);
  bottomPad->SetRightMargin(0.05);
  bottomPad->SetLogx(useLogScaleX);
  bottomPad->SetLogy(false);

  canvas->cd();
  topPad->Draw();
  topPad->cd();

  int colors[6] = { 1, 2, 3, 4, 6, 7 };
  int markerStyles[6] = { 22, 32, 20, 24, 21, 25 };

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.44, legendY0 + 0.20, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  normalizeHistogram(histogram_ref);
  histogram_ref->SetTitle("");
  histogram_ref->SetStats(false);
  histogram_ref->SetMinimum(yMin);
  histogram_ref->SetMaximum(yMax);
  histogram_ref->SetLineColor(colors[0]);
  histogram_ref->SetLineWidth(2);
  histogram_ref->SetMarkerColor(colors[0]);
  histogram_ref->SetMarkerStyle(markerStyles[0]);
  histogram_ref->Draw("e1p");
  legend->AddEntry(histogram_ref, legendEntry_ref.data(), "p");

  TAxis* xAxis_top = histogram_ref->GetXaxis();
  xAxis_top->SetTitle(xAxisTitle.data());
  xAxis_top->SetTitleOffset(xAxisOffset);
  xAxis_top->SetLabelColor(10);
  xAxis_top->SetTitleColor(10);

  TAxis* yAxis_top = histogram_ref->GetYaxis();
  yAxis_top->SetTitle(yAxisTitle.data());
  yAxis_top->SetTitleOffset(yAxisOffset);

  if ( histogram2 ) {
    normalizeHistogram(histogram2);
    histogram2->SetLineColor(colors[1]);
    histogram2->SetLineWidth(2);
    histogram2->SetMarkerColor(colors[1]);
    histogram2->SetMarkerStyle(markerStyles[1]);
    histogram2->Draw("e1psame");
    legend->AddEntry(histogram2, legendEntry2.data(), "p");
  }

  if ( histogram3 ) {
    normalizeHistogram(histogram3);
    histogram3->SetLineColor(colors[2]);
    histogram3->SetLineWidth(2);
    histogram3->SetMarkerColor(colors[2]);
    histogram3->SetMarkerStyle(markerStyles[2]);
    histogram3->Draw("e1psame");
    legend->AddEntry(histogram3, legendEntry3.data(), "p");
  }

  if ( histogram4 ) {
    normalizeHistogram(histogram4);
    histogram4->SetLineColor(colors[3]);
    histogram4->SetLineWidth(2);
    histogram4->SetMarkerColor(colors[3]);
    histogram4->SetMarkerStyle(markerStyles[3]);
    histogram4->Draw("e1psame");
    legend->AddEntry(histogram4, legendEntry4.data(), "p");
  }

  if ( histogram5 ) {
    normalizeHistogram(histogram5);
    histogram5->SetLineColor(colors[4]);
    histogram5->SetLineWidth(2);
    histogram5->SetMarkerColor(colors[4]);
    histogram5->SetMarkerStyle(markerStyles[4]);
    histogram5->Draw("e1psame");
    legend->AddEntry(histogram5, legendEntry5.data(), "p");
  }

  if ( histogram6 ) {
    normalizeHistogram(histogram6);
    histogram6->SetLineColor(colors[5]);
    histogram6->SetLineWidth(2);
    histogram6->SetMarkerColor(colors[5]);
    histogram6->SetMarkerStyle(markerStyles[5]);
    histogram6->Draw("e1psame");
    legend->AddEntry(histogram6, legendEntry6.data(), "p");
  }

  legend->Draw();

  canvas->cd();
  bottomPad->Draw();
  bottomPad->cd();

  TH1* histogram2_div_ref = 0;
  if ( histogram2 ) {
    std::string histogramName2_div_ref = std::string(histogram2->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram2_div_ref = compRatioHistogram(histogramName2_div_ref, histogram2, histogram_ref);
    histogram2_div_ref->SetTitle("");
    histogram2_div_ref->SetStats(false);
    histogram2_div_ref->SetMinimum(-0.50);
    histogram2_div_ref->SetMaximum(+0.50);

    TAxis* xAxis_bottom = histogram2_div_ref->GetXaxis();
    xAxis_bottom->SetTitle(xAxis_top->GetTitle());
    xAxis_bottom->SetLabelColor(1);
    xAxis_bottom->SetTitleColor(1);
    xAxis_bottom->SetTitleOffset(1.20);
    xAxis_bottom->SetTitleSize(0.08);
    xAxis_bottom->SetLabelOffset(0.02);
    xAxis_bottom->SetLabelSize(0.08);
    xAxis_bottom->SetTickLength(0.055);
    
    TAxis* yAxis_bottom = histogram2_div_ref->GetYaxis();
    yAxis_bottom->SetTitle("#frac{Embedding - Z/#gamma^{*} #rightarrow #tau #tau}{Z/#gamma^{*} #rightarrow #tau #tau}");
    yAxis_bottom->SetTitleOffset(0.70);
    yAxis_bottom->SetNdivisions(505);
    yAxis_bottom->CenterTitle();
    yAxis_bottom->SetTitleSize(0.08);
    yAxis_bottom->SetLabelSize(0.08);
    yAxis_bottom->SetTickLength(0.04);  
  
    histogram2_div_ref->Draw("e1p");
  }

  TH1* histogram3_div_ref = 0;
  if ( histogram3 ) {
    std::string histogramName3_div_ref = std::string(histogram3->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram3_div_ref = compRatioHistogram(histogramName3_div_ref, histogram3, histogram_ref);
    histogram3_div_ref->Draw("e1psame");
  }

  TH1* histogram4_div_ref = 0;
  if ( histogram4 ) {
    std::string histogramName4_div_ref = std::string(histogram4->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram4_div_ref = compRatioHistogram(histogramName4_div_ref, histogram4, histogram_ref);
    histogram4_div_ref->Draw("e1psame");
  }

  TH1* histogram5_div_ref = 0;
  if ( histogram5 ) {
    std::string histogramName5_div_ref = std::string(histogram5->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram5_div_ref = compRatioHistogram(histogramName5_div_ref, histogram5, histogram_ref);
    histogram5_div_ref->Draw("e1psame");
  }

  TH1* histogram6_div_ref = 0;
  if ( histogram6 ) {
    std::string histogramName6_div_ref = std::string(histogram6->GetName()).append("_div_").append(histogram_ref->GetName());
    histogram6_div_ref = compRatioHistogram(histogramName6_div_ref, histogram6, histogram_ref);
    histogram6_div_ref->Draw("e1psame");
  }

  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( useLogScaleY ) outputFileName_plot.append("_log");
  else outputFileName_plot.append("_linear");
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  //canvas->Print(std::string(outputFileName_plot).append(".pdf").data());

  delete legend;
  delete histogram2_div_ref;
  delete histogram3_div_ref;
  delete histogram4_div_ref;
  delete histogram5_div_ref;
  delete histogram6_div_ref;
  delete topPad;
  delete bottomPad;
  delete canvas;  
}
//-------------------------------------------------------------------------------

struct plotEntryType
{
  plotEntryType(TTree* tree_signal, TTree* tree_background, 
		const std::string& variableName, const std::string& treeFormula, 
		const std::string& xAxisTitle, int numBins, double xMin, double xMax, 
		double minTauPt, double maxTauPt)
    : variableName_(variableName),
      xAxisTitle_(xAxisTitle),
      numBins_(numBins),
      xMin_(xMin),
      xMax_(xMax),
      minTauPt_(minTauPt),
      maxTauPt_(maxTauPt),
      treeFormula_signal_(0),
      histogram_signal_(0),
      treeFormula_background_(0),
      histogram_background_(0)
  {    
    if      ( minTauPt_ < 0 && maxTauPt_ < 0 ) tauPtLabel_ = "allTauPt";
    else if (                  maxTauPt_ < 0 ) tauPtLabel_ = Form("tauPtGt%1.0f", minTauPt_);
    else if ( minTauPt_ < 0                  ) tauPtLabel_ = Form("tauPtLt%1.0f", maxTauPt_);
    else                                       tauPtLabel_ = Form("tauPt%1.0fto%1.0f", minTauPt_, maxTauPt_);
    std::string treeFormulaName_signal = Form("%s_%s_formula_signal", variableName.data(), tauPtLabel_.data());
    treeFormula_signal_ = new TTreeFormula(treeFormulaName_signal.data(), treeFormula.data(), tree_signal);    
    std::string histogramName_signal = Form("%s_%s_signal", variableName.data(), tauPtLabel_.data());
    histogram_signal_ = new TH1D(histogramName_signal.data(), variableName.data(), numBins, xMin, xMax);
    std::string treeFormulaName_background = Form("%s_%s_formula_background", variableName.data(), tauPtLabel_.data());
    treeFormula_background_ = new TTreeFormula(treeFormulaName_background.data(), treeFormula.data(), tree_background);
    std::string histogramName_background = Form("%s_%s_background", variableName.data(), tauPtLabel_.data());
    histogram_background_ = new TH1D(histogramName_background.data(), variableName.data(), numBins, xMin, xMax);
  }
  ~plotEntryType()
  {
    delete treeFormula_signal_;
    delete histogram_signal_;
    delete treeFormula_background_;
    delete histogram_background_;
  }
  void fill_signal(double tauPt, double evtWeight)
  {
    fill_with_check(histogram_signal_, treeFormula_signal_, tauPt, evtWeight);
  }
  void fill_background(double tauPt, double evtWeight)
  {
    fill_with_check(histogram_background_, treeFormula_background_, tauPt, evtWeight);
  }
  void fill_with_check(TH1* histogram, TTreeFormula* treeFormula, double tauPt, double evtWeight)
  {
    if ( (minTauPt_ < 0. || tauPt > minTauPt_) && 
	 (maxTauPt_ < 0. || tauPt < maxTauPt_) ) {
      //std::cout << "evaluating treeFormula " << treeFormula->GetName() << ", formula = " << treeFormula->GetTitle() << ": value = " << treeFormula->EvalInstance() << std::endl;
      histogram->Fill(treeFormula->EvalInstance(), evtWeight);
    }
  }
  std::string variableName_;
  std::string tauPtLabel_; 
  std::string xAxisTitle_;
  int numBins_;
  double xMin_;
  double xMax_;
  double minTauPt_;
  double maxTauPt_;
  TTreeFormula* treeFormula_signal_;
  TH1* histogram_signal_;
  TTreeFormula* treeFormula_background_;
  TH1* histogram_background_;
};

struct mvaInputVariableType
{
  mvaInputVariableType(TTree* tree_signal, TTree* tree_background, const std::string& variableName, const std::string& treeFormula) 
    : tree_signal_(tree_signal),
      tree_background_(tree_background)
  {
    std::string treeFormulaName_signal = Form("mvaInputVariable%s_formula_signal", variableName.data());
    treeFormula_signal_ = new TTreeFormula(treeFormulaName_signal.data(), treeFormula.data(), tree_signal);    
    std::string treeFormulaName_background = Form("mvaInputVariable%s_formula_background", variableName.data());
    treeFormula_background_ = new TTreeFormula(treeFormulaName_background.data(), treeFormula.data(), tree_background);
  }
  ~mvaInputVariableType()
  {
    delete treeFormula_signal_;
    delete treeFormula_background_;
  }
  void update(const TTree* tree)
  {
    if      ( tree == tree_signal_     ) value_ = treeFormula_signal_->EvalInstance();
    else if ( tree == tree_background_ ) value_ = treeFormula_background_->EvalInstance();
    else assert(0);
  }
  const TTree* tree_signal_;
  TTreeFormula* treeFormula_signal_;
  const TTree* tree_background_;
  TTreeFormula* treeFormula_background_;
  Float_t value_;
};

void plotTauIdMVAInputVariables()
{
//--- stop ROOT from keeping references to all histograms
  TH1::AddDirectory(false);

//--- suppress the output canvas 
  gROOT->SetBatch(true);

  gSystem->Load("libTreePlayer");

  TString inputFilePath = "/data2/veelken/CMSSW_5_3_x/Ntuples/tauIdMVATraining/v0_4/";
  inputFilePath.Append("user/veelken/CMSSW_5_3_x/Ntuples/tauIdMVATraining/v0_4/");
 
  TString inputFileName = "tauIdMVATrainingNtuple_*.root";

  std::string mvaFileName = "../test/weights/mvaIsolation3HitsDeltaR04opt1_BDTG.weights.xml";
  std::vector<std::string> mvaInputVariables_string;
  mvaInputVariables_string.push_back("recTauPt");
  //mvaInputVariables_string.push_back("TMath::Log(TMath::Max(1., recTauPt))");
  mvaInputVariables_string.push_back("TMath::Abs(recTauEta)");
  mvaInputVariables_string.push_back("TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsChargedIsoPtSum))");
  mvaInputVariables_string.push_back("TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsNeutralIsoPtSum))");
  mvaInputVariables_string.push_back("TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsPUcorrPtSum))");
  mvaInputVariables_string.push_back("recTauDecayMode");
  mvaInputVariables_string.push_back("numOfflinePrimaryVertices");
  double mvaCut = 0.95;

  //-------------------------------------------------------------------------------------------------
  std::vector<std::string> samples_signal;
  samples_signal.push_back("ZplusJets_madgraph");
  samples_signal.push_back("ggHiggs110toTauTau");
  samples_signal.push_back("vbfHiggs110toTauTau");
  samples_signal.push_back("ggHiggs120toTauTau");
  samples_signal.push_back("vbfHiggs120toTauTau");
  samples_signal.push_back("ggHiggs130toTauTau");
  samples_signal.push_back("vbfHiggs130toTauTau");
  samples_signal.push_back("ggHiggs140toTauTau");
  samples_signal.push_back("vbfHiggs140toTauTau");
  samples_signal.push_back("ggA160toTauTau");
  samples_signal.push_back("bbA160toTauTau");
  samples_signal.push_back("ggA180toTauTau");
  samples_signal.push_back("bbA180toTauTau");
  samples_signal.push_back("ggA200toTauTau");
  samples_signal.push_back("bbA200toTauTau");
  samples_signal.push_back("ggA250toTauTau");
  samples_signal.push_back("bbA250toTauTau");
  samples_signal.push_back("ggA300toTauTau");
  samples_signal.push_back("bbA300toTauTau");
  samples_signal.push_back("ggA350toTauTau");
  samples_signal.push_back("bbA350toTauTau");
  samples_signal.push_back("ggA400toTauTau");
  samples_signal.push_back("bbA400toTauTau");
  samples_signal.push_back("ggA450toTauTau");
  samples_signal.push_back("bbA450toTauTau");
  samples_signal.push_back("ggA500toTauTau");
  samples_signal.push_back("bbA500toTauTau");
  samples_signal.push_back("ggA600toTauTau");
  samples_signal.push_back("bbA600toTauTau");
  samples_signal.push_back("ggA700toTauTau");
  samples_signal.push_back("bbA700toTauTau");
  samples_signal.push_back("ggA800toTauTau");
  samples_signal.push_back("bbA800toTauTau");
  samples_signal.push_back("ggA900toTauTau");
  samples_signal.push_back("bbA900toTauTau");
  samples_signal.push_back("ggA1000toTauTau");
  samples_signal.push_back("bbA1000toTauTau");
  samples_signal.push_back("Zprime750toTauTau");
  samples_signal.push_back("Zprime1000toTauTau");
  samples_signal.push_back("Zprime1250toTauTau");
  samples_signal.push_back("Zprime1500toTauTau");
  samples_signal.push_back("Zprime1750toTauTau");
  samples_signal.push_back("Zprime2000toTauTau");
  samples_signal.push_back("Zprime2250toTauTau");
  samples_signal.push_back("Zprime2500toTauTau");

  TChain* tree_signal = new TChain("tauIdMVATrainingNtupleProducer/tauIdMVATrainingNtuple");
  for ( std::vector<std::string>::const_iterator sample = samples_signal.begin();
	sample != samples_signal.end(); ++sample ) {
    tree_signal->Add(TString(inputFilePath).Append(*sample).Append("/").Append(inputFileName));
  }  
  std::cout << "signal Tree contains " << tree_signal->GetEntries() << " Entries in " << tree_signal->GetListOfFiles()->GetEntries() << " files." << std::endl;
  // CV: need to call TChain::LoadTree before processing first event 
  //     in order to prevent ROOT causing a segmentation violation,
  //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=10062
  tree_signal->LoadTree(0);
  //-------------------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------------------
  std::vector<std::string> samples_background;
  samples_background.push_back("QCDjetsFlatPt15to3000");
  samples_background.push_back("PPmuXptGt20Mu15");
  samples_background.push_back("WplusJets_madgraph");
  
  TChain* tree_background = new TChain("tauIdMVATrainingNtupleProducer/tauIdMVATrainingNtuple");
  for ( std::vector<std::string>::const_iterator sample = samples_background.begin();
	sample != samples_background.end(); ++sample ) {
    tree_background->Add(TString(inputFilePath).Append(*sample).Append("/").Append(inputFileName));
  }
  std::cout << "background Tree contains " << tree_background->GetEntries() << " Entries in " << tree_background->GetListOfFiles()->GetEntries() << " files." << std::endl;
  // CV: need to call TChain::LoadTree before processing first event 
  //     in order to prevent ROOT causing a segmentation violation,
  //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=10062
  tree_background->LoadTree(0);
  //-------------------------------------------------------------------------------------------------

  std::vector<mvaInputVariableType*> mvaInputVariables;
  TMVA::Reader* mva = 0;
  if ( mvaFileName != "" ) {
    TMVA::Tools::Instance();
    mva = new TMVA::Reader("!V:!Silent");
    int idx = 0;
    for ( std::vector<std::string>::const_iterator mvaInputVariable_string = mvaInputVariables_string.begin();
	  mvaInputVariable_string != mvaInputVariables_string.end(); ++mvaInputVariable_string ) {
      mvaInputVariableType* mvaInputVariable = new mvaInputVariableType(tree_signal, tree_background, Form("%i", idx), *mvaInputVariable_string);
      mva->AddVariable(*mvaInputVariable_string, &mvaInputVariable->value_);
      mvaInputVariables.push_back(mvaInputVariable);
      ++idx;
    }
    mva->BookMVA("BDTG", mvaFileName.data());
  }

  typedef std::pair<double, double> pdouble;
  std::vector<pdouble> ptBins;
  ptBins.push_back(pdouble(  -1.,   200.));
  ptBins.push_back(pdouble( 200.,   400.));
  ptBins.push_back(pdouble( 400.,   600.));
  ptBins.push_back(pdouble( 600.,   900.));
  ptBins.push_back(pdouble( 900.,  1200.));
  ptBins.push_back(pdouble(1200.,    -1.));
  ptBins.push_back(pdouble(  -1.,    -1.)); 
  
  std::vector<plotEntryType*> plots;
  for ( std::vector<pdouble>::const_iterator ptBin = ptBins.begin();
	ptBin != ptBins.end(); ++ptBin ) {
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "recTauPt", 
      "recTauPt", "P_{T} / GeV", 100, 0., 2000., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "recTauEta", 
      "recTauEta", "#eta", 50, -2.5, +2.5, ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "recTauM", 
      "recTauM", "M / GeV", 50, 0., 5., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "recTauDecayMode", 
      "recTauDecayMode", "Decay Mode", 15, -0.5, 14.5, ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "leadPFCandPtDivRecTauPt",   
      "leadPFCandPt/recTauPt", "P_{T} / GeV", 100, 0., 2., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "leadPFChargedHadrCandPtDivRecTauPt",
      "leadPFChargedHadrCandPt/recTauPt", "P_{T} / GeV", 100, 0., 2., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "chargedHadron1PtDivRecTauPt",
      "chargedHadron1Pt/recTauPt", "P_{T} / GeV", 100, 0., 2., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "chargedHadron2PtDivRecTauPt", 
      "chargedHadron2Pt/recTauPt", "P_{T} / GeV", 100, 0., 2., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "chargedHadron3PtDivRecTauPt",
      "chargedHadron3Pt/recTauPt", "P_{T} / GeV", 100, 0., 2., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "piZero1PtDivRecTauPt",      
      "piZero1Pt/recTauPt", "P_{T} / GeV", 100,  0., 2., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "piZero1NumPFGammas", 
      "piZero1NumPFGammas", "N_{#gamma}", 20, -0.5, 19.5, ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "piZero1NumPFElectrons",   
      "piZero1NumPFElectrons", "N_{e}", 5, -0.5, 4.5, ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "piZero1MaxDeltaEta",  
      "piZero1MaxDeltaEta", "#Delta#eta_{max}", 50, 0., 0.50, ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "piZero1MaxDeltaPhi", 
      "piZero1MaxDeltaPhi", "#Delta#phi_{max}", 50, 0., 0.10, ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "piZero1M",   
      "piZero1M", "M / GeV", 100, 0., 1., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR04PtThresholdsLoose3HitsChargedIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsChargedIsoPtSum))", "log(I_{charged} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR04PtThresholdsLoose3HitsNeutralIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsNeutralIsoPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR04PtThresholdsLoose3HitsPUcorrPtSum",   
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose3HitsPUcorrPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR04PtThresholdsLoose8HitsChargedIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose8HitsChargedIsoPtSum))", "log(I_{charged} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR04PtThresholdsLoose8HitsNeutralIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose8HitsNeutralIsoPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR04PtThresholdsLoose8HitsPUcorrPtSum",   
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR04PtThresholdsLoose8HitsPUcorrPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))", "log(I_{charged} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum",   
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsPUcorrPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR05PtThresholdsLoose8HitsChargedIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose8HitsChargedIsoPtSum))", "log(I_{charged} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR05PtThresholdsLoose8HitsNeutralIsoPtSum", 
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose8HitsNeutralIsoPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "logTauIsoDeltaR05PtThresholdsLoose8HitsPUcorrPtSum",   
      "TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose8HitsPUcorrPtSum))", "log(I_{neutral} / GeV)", 120, -5., +7., ptBin->first, ptBin->second));    
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "numOfflinePrimaryVertices",   
      "numOfflinePrimaryVertices", "N_{vtx}", 60, -0.5, 59.5, ptBin->first, ptBin->second));
    plots.push_back(new plotEntryType(
      tree_signal, tree_background, "numPileUp", 
      "numPileUp", "N_{PU}", 60, -0.5, 59.5, ptBin->first, ptBin->second));
  }

  if ( inputFilePath.Length() > 0 && !inputFilePath.EndsWith("/") ) inputFilePath.Append("/");

  //-------------------------------------------------------------------------------------------------
  Float_t recTauPt_signal;
  tree_signal->SetBranchAddress("recTauPt", &recTauPt_signal);

  Float_t evtWeight_signal;
  tree_signal->SetBranchAddress("evtWeight", &evtWeight_signal);
  
  const int maxEvents_signal = 1000000;
  
  int currentTreeNumber_signal = tree_signal->GetTreeNumber();

  int numEntries_signal = tree_signal->GetEntries();
  for ( int iEntry = 0; iEntry < numEntries_signal && (iEntry < maxEvents_signal || maxEvents_signal == -1); ++iEntry ) {
    if ( iEntry > 0 && (iEntry % 10000) == 0 ) {
      std::cout << "processing signal Entry " << iEntry << std::endl;
    }
    
    tree_signal->GetEntry(iEntry);
    
    // CV: need to call TTreeFormula::UpdateFormulaLeaves whenever input files changes in TChain
    //     in order to prevent ROOT causing a segmentation violation,
    //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=481
    if ( tree_signal->GetTreeNumber() != currentTreeNumber_signal ) {
      for ( std::vector<plotEntryType*>::iterator plot = plots.begin();
	    plot != plots.end(); ++plot ) {
	(*plot)->treeFormula_signal_->UpdateFormulaLeaves();
      }
      for ( std::vector<mvaInputVariableType*>::iterator mvaInputVariable = mvaInputVariables.begin();
	    mvaInputVariable != mvaInputVariables.end(); ++mvaInputVariable ) {
	(*mvaInputVariable)->treeFormula_signal_->UpdateFormulaLeaves();
      }
      currentTreeNumber_signal = tree_signal->GetTreeNumber();
    }

    if ( mva ) {
      for ( std::vector<mvaInputVariableType*>::iterator mvaInputVariable = mvaInputVariables.begin();
	    mvaInputVariable != mvaInputVariables.end(); ++mvaInputVariable ) {
	(*mvaInputVariable)->update(tree_signal);
      }
      double mvaOutput = mva->EvaluateMVA("BDTG");
      if ( mvaOutput < mvaCut ) continue;
    }

    for ( std::vector<plotEntryType*>::iterator plot = plots.begin();
	  plot != plots.end(); ++plot ) {
      (*plot)->fill_signal(recTauPt_signal, evtWeight_signal);
    }
  }

  delete tree_signal;
  //-------------------------------------------------------------------------------------------------

  //-------------------------------------------------------------------------------------------------
  Float_t recTauPt_background;
  tree_background->SetBranchAddress("recTauPt", &recTauPt_background);

  Float_t evtWeight_background;
  tree_background->SetBranchAddress("evtWeight", &evtWeight_background);

  const int maxEvents_background = -1;

  int currentTreeNumber_background = tree_background->GetTreeNumber();

  int numEntries_background = tree_background->GetEntries();
  for ( int iEntry = 0; iEntry < numEntries_background && (iEntry < maxEvents_background || maxEvents_background == -1); ++iEntry ) {
    if ( iEntry > 0 && (iEntry % 10000) == 0 ) {
      std::cout << "processing background Entry " << iEntry << std::endl;
    }
    
    tree_background->GetEntry(iEntry);
    
    // CV: need to call TTreeFormula::UpdateFormulaLeaves whenever input files changes in TChain
    //     in order to prevent ROOT causing a segmentation violation,
    //     cf. http://root.cern.ch/phpBB3/viewtopic.php?t=481
    if ( tree_background->GetTreeNumber() != currentTreeNumber_background ) {
      for ( std::vector<plotEntryType*>::iterator plot = plots.begin();
	    plot != plots.end(); ++plot ) {
	(*plot)->treeFormula_background_->UpdateFormulaLeaves();
      }
      for ( std::vector<mvaInputVariableType*>::iterator mvaInputVariable = mvaInputVariables.begin();
	    mvaInputVariable != mvaInputVariables.end(); ++mvaInputVariable ) {
	(*mvaInputVariable)->treeFormula_background_->UpdateFormulaLeaves();
      }
      currentTreeNumber_background = tree_background->GetTreeNumber();
    }

    if ( mva ) {
      for ( std::vector<mvaInputVariableType*>::iterator mvaInputVariable = mvaInputVariables.begin();
	    mvaInputVariable != mvaInputVariables.end(); ++mvaInputVariable ) {
	(*mvaInputVariable)->update(tree_background);
      }
      double mvaOutput = mva->EvaluateMVA("BDTG");
      if ( mvaOutput < mvaCut ) continue;
    }
    
    for ( std::vector<plotEntryType*>::iterator plot = plots.begin();
	  plot != plots.end(); ++plot ) {
      (*plot)->fill_background(recTauPt_background, evtWeight_background);
    }
  }

  delete tree_background;
  //-------------------------------------------------------------------------------------------------

  for ( std::vector<plotEntryType*>::iterator plot = plots.begin();
	plot != plots.end(); ++plot ) {
    if ( (*plot)->histogram_signal_->Integral()     >= 100 &&
	 (*plot)->histogram_background_->Integral() >= 100 ) {
      std::cout << "drawing plot = " << (*plot)->variableName_ << ":"
		<< " integral(" << (*plot)->histogram_signal_->GetName() << " = " << (*plot)->histogram_signal_->Integral() << ","
		<< " integral(" << (*plot)->histogram_background_->GetName() << " = " << (*plot)->histogram_background_->Integral() << std::endl;      
      bool useLogScaleX = ( (*plot)->numBins_ > 100 ) ? true : false;
      showDistribution(800, 900,
		       (*plot)->histogram_signal_, "Signal",
		       (*plot)->histogram_background_, "Background",
		       0, "",
		       0, "",
		       0, "",
		       0, "",
		       useLogScaleX, (*plot)->xAxisTitle_, 1.2,
		       false, 0., 1.2, "a.u", 1.2,
		       0.50, 0.74,
		       Form("plots/plotTauIdMVAInputVariables_%s_%s_linear.png", (*plot)->variableName_.data(), (*plot)->tauPtLabel_.data()));
      showDistribution(800, 900,
		       (*plot)->histogram_signal_, "Signal",
		       (*plot)->histogram_background_, "Background",
		       0, "",
		       0, "",
		       0, "",
		       0, "",
		       useLogScaleX, (*plot)->xAxisTitle_, 1.2,
		       true, 1.e-5, 1.e+2, "a.u", 1.2,
		       0.50, 0.74,
		       Form("plots/plotTauIdMVAInputVariables_%s_%s_log.png", (*plot)->variableName_.data(), (*plot)->tauPtLabel_.data()));
    } else {
      std::cout << "plot = " << (*plot)->variableName_ << " has not enough Event statistics:"
		<< " integral(" << (*plot)->histogram_signal_->GetName() << " = " << (*plot)->histogram_signal_->Integral() << ","
		<< " integral(" << (*plot)->histogram_background_->GetName() << " = " << (*plot)->histogram_background_->Integral() << " --> distributions will NOT be plotted !!" << std::endl;
    }
  }

  TMVA::Tools::DestroyInstance();
  delete mva;
  for ( std::vector<mvaInputVariableType*>::iterator it = mvaInputVariables.begin();
	it != mvaInputVariables.end(); ++it ) {
    delete (*it);
  }
  for ( std::vector<plotEntryType*>::iterator it = plots.begin();
	it != plots.end(); ++it ) {
    delete (*it);
  }
}
