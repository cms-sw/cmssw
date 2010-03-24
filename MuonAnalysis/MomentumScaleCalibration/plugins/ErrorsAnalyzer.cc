#ifndef ERRORSANALYZER_CC
#define ERRORSANALYZER_CC

#include "ErrorsAnalyzer.h"

ErrorsAnalyzer::ErrorsAnalyzer(const edm::ParameterSet& iConfig) :
  treeFileName_( iConfig.getParameter<string>("InputFileName") ),
  resolFitType_( iConfig.getParameter<int>("ResolFitType") ),
  maxEvents_( iConfig.getParameter<int>("MaxEvents") ),
  outputFileName_( iConfig.getParameter<string>("OutputFileName") ),
  ptBins_( iConfig.getParameter<int>("PtBins") ),
  etaBins_( iConfig.getParameter<int>("EtaBins") ),
  debug_( iConfig.getParameter<bool>("Debug") )
{
  parameters_ = iConfig.getParameter<vector<double> >("Parameters");
  errors_ = iConfig.getParameter<vector<double> >("Errors");
  errorFactors_ = iConfig.getParameter<vector<int> >("ErrorFactors");

  if( (parameters_.size() != errors_.size()) || (parameters_.size() != errorFactors_.size()) ) {
    cout << "Error: parameters and errors have different number of values" << endl;
    exit(1);
  }

  fillValueError();

  sigmaPtVsPt_ = new TProfile("sigmaPtVsPtProfile", "sigmaPtVsPt", ptBins_, 0., 20.);
  sigmaPtVsPtPlusErr_ = new TProfile("sigmaPtVsPtPlusErrProfile", "sigmaPtVsPtPlusErr", ptBins_, 0., 20.);
  sigmaPtVsPtMinusErr_ = new TProfile("sigmaPtVsPtMinusErrProfile", "sigmaPtVsPtMinusErr", ptBins_, 0., 20.);

  sigmaPtVsEta_ = new TProfile("sigmaPtVsEtaProfile", "sigmaPtVsEta", etaBins_, -3., 3.);
  sigmaPtVsEtaPlusErr_ = new TProfile("sigmaPtVsEtaPlusErrProfile", "sigmaPtVsEtaPlusErr", etaBins_, -3., 3.);
  sigmaPtVsEtaMinusErr_ = new TProfile("sigmaPtVsEtaMinusErrProfile", "sigmaPtVsEtaMinusErr", etaBins_, -3., 3.);
}

void ErrorsAnalyzer::fillValueError()
{
  valuePlusError_.resize(parameters_.size());
  valueMinusError_.resize(parameters_.size());

  vector<double>::const_iterator parIt = parameters_.begin();
  vector<double>::const_iterator errIt = errors_.begin();
  vector<int>::const_iterator errFactorIt = errorFactors_.begin();
  int i=0;
  for( ; parIt != parameters_.end(); ++parIt, ++errIt, ++errFactorIt, ++i ) {
    valuePlusError_[i] = *parIt + (*errIt)*(*errFactorIt);
    valueMinusError_[i] = *parIt - (*errIt)*(*errFactorIt);
  }
}

ErrorsAnalyzer::~ErrorsAnalyzer()
{
  gROOT->SetStyle("Plain");

  fillHistograms();

  TFile * outputFile = new TFile(outputFileName_, "RECREATE");

  drawHistograms(sigmaPtVsEta_, sigmaPtVsEtaPlusErr_, sigmaPtVsEtaMinusErr_, "Eta");
  drawHistograms(sigmaPtVsPt_, sigmaPtVsPtPlusErr_, sigmaPtVsPtMinusErr_, "Pt");

  outputFile->Write();
  outputFile->Close();
}

void ErrorsAnalyzer::drawHistograms(const TProfile * histo, const TProfile * histoPlusErr,
				    const TProfile * histoMinusErr, const TString & type)
{
  TH1D * sigmaPtVsEtaTH1D = new TH1D("sigmaPtVs"+type, "sigmaPtVs"+type, histo->GetNbinsX(),
				     histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  TH1D * sigmaPtVsEtaPlusErrTH1D = new TH1D("sigmaPtVs"+type+"PlusErr", "sigmaPtVs"+type+"PlusErr", histo->GetNbinsX(),
					    histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  TH1D * sigmaPtVsEtaMinusErrTH1D = new TH1D("sigmaPtVs"+type+"MinusErr", "sigmaPtVs"+type+"MinusErr", histo->GetNbinsX(),
					     histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  TCanvas * canvas = new TCanvas("canvas"+type, "canvas"+type, 1000, 800);
  for( int iBin = 1; iBin <= histo->GetNbinsX(); ++iBin ) {
    sigmaPtVsEtaTH1D->SetBinContent(iBin, histo->GetBinContent(iBin));
    sigmaPtVsEtaPlusErrTH1D->SetBinContent(iBin, histoPlusErr->GetBinContent(iBin));
    sigmaPtVsEtaMinusErrTH1D->SetBinContent(iBin, histoMinusErr->GetBinContent(iBin));
  }
  int numBins = sigmaPtVsEtaTH1D->GetNbinsX();
  // Draw TGraph with asymmetric errors
  Double_t * values = sigmaPtVsEtaTH1D->GetArray();
  Double_t * valuesPlus = sigmaPtVsEtaPlusErrTH1D->GetArray();
  Double_t * valuesMinus = sigmaPtVsEtaMinusErrTH1D->GetArray();
  double * posErrors = new double[numBins];
  double * negErrors = new double[numBins];

  TGraphAsymmErrors * graphAsymmErrors = new TGraphAsymmErrors(sigmaPtVsEtaTH1D);
  TGraph * graph = new TGraph(sigmaPtVsEtaTH1D);

  for( int i=1; i<=numBins; ++i ) {
    // cout << "filling " << i << endl;
    posErrors[i-1] = valuesPlus[i] - values[i];
    if( valuesMinus[i-1] < 0 ) negErrors[i] = 0;
    else negErrors[i-1] = values[i] - valuesMinus[i];

    graphAsymmErrors->SetPointEYlow(i, negErrors[i-1]);
    graphAsymmErrors->SetPointEYhigh(i, posErrors[i-1]);
  }

  canvas->Draw();

  graphAsymmErrors->SetTitle("");
  graphAsymmErrors->SetFillColor(kGray);
  graphAsymmErrors->Draw("A2");
  TString title(type);
  if( type == "Eta" ) title = "#eta";
  graphAsymmErrors->GetXaxis()->SetTitle(title);
  graphAsymmErrors->GetYaxis()->SetTitle("#sigmaPt/Pt");
  graph->Draw();

  //  graph->SetLineColor(kGray);
  //  graph->SetMarkerColor(kBlack);
  //  graph->Draw("AP");

//   if( debug_ ) {
//     sigmaPtVsEtaPlusErrTH1D->SetLineColor(kRed);
//     sigmaPtVsEtaMinusErrTH1D->SetLineColor(kBlue);
//   }
//   else {
//     sigmaPtVsEtaPlusErrTH1D->SetFillColor(kGray);
//     sigmaPtVsEtaPlusErrTH1D->SetLineColor(kWhite);
//     sigmaPtVsEtaMinusErrTH1D->SetFillColor(kWhite);
//     sigmaPtVsEtaMinusErrTH1D->SetLineColor(kWhite);
//   }
//   sigmaPtVsEtaPlusErrTH1D->Draw();
//   sigmaPtVsEtaTH1D->Draw("SAME");
//   sigmaPtVsEtaMinusErrTH1D->Draw("SAME");

  sigmaPtVsEtaPlusErrTH1D->Write();
  sigmaPtVsEtaTH1D->Write();
  sigmaPtVsEtaMinusErrTH1D->Write();

  sigmaPtVsEtaPlusErr_->Write();
  sigmaPtVsEta_->Write();
  sigmaPtVsEtaMinusErr_->Write();

  canvas->Write();
}

// ------------ method called to for each event  ------------
void ErrorsAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

void ErrorsAnalyzer::fillHistograms()
{
  cout << "Reading muon pairs from Root Tree in " << treeFileName_ << endl;
  RootTreeHandler rootTreeHandler;

  typedef vector<pair<lorentzVector,lorentzVector> > MuonPairVector;
  MuonPairVector savedPair;
  rootTreeHandler.readTree(maxEvents_, treeFileName_, &savedPair);
  // rootTreeHandler.readTree(maxEvents, inputRootTreeFileName_, &savedPair, &(MuScleFitUtils::genPair));

  resolutionFunctionBase<vector<double> > * resolutionFunctionForVec = resolutionFunctionVecService( resolFitType_ );

  // Loop on all the pairs
  unsigned int i = 0;
  MuonPairVector::iterator it = savedPair.begin();
  cout << "Starting loop on " << savedPair.size() << " muons" << endl;
  for( ; it != savedPair.end(); ++it, ++i ) {
    cout << "Muon pair number " << i << endl;
    double pt1 = it->first.pt();
    double eta1 = it->first.eta();
    double pt2 = it->second.pt();
    double eta2 = it->second.eta();

    double sigmaPt1 = resolutionFunctionForVec->sigmaPt( pt1,eta1,parameters_ );
    double sigmaPt2 = resolutionFunctionForVec->sigmaPt( pt2,eta2,parameters_ );
    double sigmaPtPlusErr1 = resolutionFunctionForVec->sigmaPt( pt1,eta1,valuePlusError_ );
    double sigmaPtPlusErr2 = resolutionFunctionForVec->sigmaPt( pt2,eta2,valuePlusError_ );
    double sigmaPtMinusErr1 = resolutionFunctionForVec->sigmaPt( pt1,eta1,valueMinusError_ );
    double sigmaPtMinusErr2 = resolutionFunctionForVec->sigmaPt( pt2,eta2,valueMinusError_ );

    sigmaPtVsPt_->Fill(pt1, sigmaPt1);
    sigmaPtVsPt_->Fill(pt2, sigmaPt2);
    sigmaPtVsPtPlusErr_->Fill(pt1, sigmaPtPlusErr1);
    sigmaPtVsPtPlusErr_->Fill(pt2, sigmaPtPlusErr2);
    sigmaPtVsPtMinusErr_->Fill(pt1, sigmaPtMinusErr1);
    sigmaPtVsPtMinusErr_->Fill(pt2, sigmaPtMinusErr2);

    sigmaPtVsEta_->Fill(eta1, sigmaPt1);
    sigmaPtVsEta_->Fill(eta2, sigmaPt2);
    sigmaPtVsEtaPlusErr_->Fill(eta1, sigmaPtPlusErr1);
    sigmaPtVsEtaPlusErr_->Fill(eta2, sigmaPtPlusErr2);
    sigmaPtVsEtaMinusErr_->Fill(eta1, sigmaPtMinusErr1);
    sigmaPtVsEtaMinusErr_->Fill(eta2, sigmaPtMinusErr2);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ErrorsAnalyzer);

#endif
