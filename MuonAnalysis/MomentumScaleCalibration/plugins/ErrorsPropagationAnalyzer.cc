#ifndef ERRORSANALYZER_CC
#define ERRORSANALYZER_CC

#include "ErrorsPropagationAnalyzer.h"

ErrorsPropagationAnalyzer::ErrorsPropagationAnalyzer(const edm::ParameterSet& iConfig) :
  treeFileName_( iConfig.getParameter<std::string>("InputFileName") ),
  resolFitType_( iConfig.getParameter<int>("ResolFitType") ),
  maxEvents_( iConfig.getParameter<int>("MaxEvents") ),
  outputFileName_( iConfig.getParameter<std::string>("OutputFileName") ),
  ptBins_( iConfig.getParameter<int>("PtBins") ),
  ptMin_( iConfig.getParameter<double>("PtMin") ),
  ptMax_( iConfig.getParameter<double>("PtMax") ),
  etaBins_( iConfig.getParameter<int>("EtaBins") ),
  etaMin_( iConfig.getParameter<double>("EtaMin") ),
  etaMax_( iConfig.getParameter<double>("EtaMax") ),
  debug_( iConfig.getParameter<bool>("Debug") ),
  ptMinCut_( iConfig.getUntrackedParameter<double>("PtMinCut", 0.) ),
  ptMaxCut_( iConfig.getUntrackedParameter<double>("PtMaxCut", 999999.) ),
  etaMinCut_( iConfig.getUntrackedParameter<double>("EtaMinCut", 0.) ),
  etaMaxCut_( iConfig.getUntrackedParameter<double>("EtaMaxCut", 100.) )
{
  parameters_ = iConfig.getParameter<std::vector<double> >("Parameters");
  errors_ = iConfig.getParameter<std::vector<double> >("Errors");
  errorFactors_ = iConfig.getParameter<std::vector<int> >("ErrorFactors");

  if( (parameters_.size() != errors_.size()) || (parameters_.size() != errorFactors_.size()) ) {
    std::cout << "Error: parameters and errors have different number of values" << std::endl;
    exit(1);
  }

  fillValueError();

  sigmaPtVsPt_ = new TProfile("sigmaPtVsPtProfile", "sigmaPtVsPt", ptBins_, ptMin_, ptMax_);
  sigmaPtVsPtPlusErr_ = new TProfile("sigmaPtVsPtPlusErrProfile", "sigmaPtVsPtPlusErr", ptBins_, ptMin_, ptMax_);
  sigmaPtVsPtMinusErr_ = new TProfile("sigmaPtVsPtMinusErrProfile", "sigmaPtVsPtMinusErr", ptBins_, ptMin_, ptMax_);

  sigmaPtVsEta_ = new TProfile("sigmaPtVsEtaProfile", "sigmaPtVsEta", etaBins_, etaMin_, etaMax_);
  sigmaPtVsEtaPlusErr_ = new TProfile("sigmaPtVsEtaPlusErrProfile", "sigmaPtVsEtaPlusErr", etaBins_, etaMin_, etaMax_);
  sigmaPtVsEtaMinusErr_ = new TProfile("sigmaPtVsEtaMinusErrProfile", "sigmaPtVsEtaMinusErr", etaBins_, etaMin_, etaMax_);

  sigmaMassVsPt_ = new TProfile("sigmaMassVsPtProfile", "sigmaMassVsPt", ptBins_, ptMin_, ptMax_);
  sigmaMassVsPtPlusErr_ = new TProfile("sigmaMassVsPtPlusErrProfile", "sigmaMassVsPtPlusErr", ptBins_, ptMin_, ptMax_);
  sigmaMassVsPtMinusErr_ = new TProfile("sigmaMassVsPtMinusErrProfile", "sigmaMassVsPtMinusErr", ptBins_, ptMin_, ptMax_);

  sigmaMassVsEta_ = new TProfile("sigmaMassVsEtaProfile", "sigmaMassVsEta", etaBins_, etaMin_, etaMax_);
  sigmaMassVsEtaPlusErr_ = new TProfile("sigmaMassVsEtaPlusErrProfile", "sigmaMassVsEtaPlusErr", etaBins_, etaMin_, etaMax_);
  sigmaMassVsEtaMinusErr_ = new TProfile("sigmaMassVsEtaMinusErrProfile", "sigmaMassVsEtaMinusErr", etaBins_, etaMin_, etaMax_);

  sigmaPtVsPtDiff_ = new TProfile("sigmaPtVsPtDiffProfile", "sigmaPtVsPtDiff", ptBins_, ptMin_, ptMax_);
  sigmaPtVsEtaDiff_ = new TProfile("sigmaPtVsEtaDiffProfile", "sigmaPtVsEtaDiff", etaBins_, etaMin_, etaMax_);
}

void ErrorsPropagationAnalyzer::fillValueError()
{
  valuePlusError_.resize(parameters_.size());
  valueMinusError_.resize(parameters_.size());

  std::vector<double>::const_iterator parIt = parameters_.begin();
  std::vector<double>::const_iterator errIt = errors_.begin();
  std::vector<int>::const_iterator errFactorIt = errorFactors_.begin();
  int i=0;
  for( ; parIt != parameters_.end(); ++parIt, ++errIt, ++errFactorIt, ++i ) {
    valuePlusError_[i] = *parIt + (*errIt)*(*errFactorIt);
    valueMinusError_[i] = *parIt - (*errIt)*(*errFactorIt);
  }
}

ErrorsPropagationAnalyzer::~ErrorsPropagationAnalyzer()
{
  gROOT->SetStyle("Plain");

  fillHistograms();

  TFile * outputFile = new TFile(outputFileName_, "RECREATE");

  drawHistograms(sigmaPtVsEta_, sigmaPtVsEtaPlusErr_, sigmaPtVsEtaMinusErr_, "sigmaPtVsEta");
  drawHistograms(sigmaPtVsPt_, sigmaPtVsPtPlusErr_, sigmaPtVsPtMinusErr_, "sigmaPtVsPt");

  drawHistograms(sigmaMassVsEta_, sigmaMassVsEtaPlusErr_, sigmaMassVsEtaMinusErr_, "sigmaMassVsEta");
  drawHistograms(sigmaMassVsPt_, sigmaMassVsPtPlusErr_, sigmaMassVsPtMinusErr_, "sigmaMassVsPt");

  sigmaPtVsPtDiff_->Write();
  sigmaPtVsEtaDiff_->Write();  

  outputFile->Write();
  outputFile->Close();
}

void ErrorsPropagationAnalyzer::drawHistograms(const TProfile * histo, const TProfile * histoPlusErr,
				    const TProfile * histoMinusErr, const TString & type)
{
  TH1D * sigmaPtVsEtaTH1D = new TH1D(type, type, histo->GetNbinsX(),
				     histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  TH1D * sigmaPtVsEtaPlusErrTH1D = new TH1D(type+"PlusErr", type+"PlusErr", histo->GetNbinsX(),
					    histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  TH1D * sigmaPtVsEtaMinusErrTH1D = new TH1D(type+"MinusErr", type+"MinusErr", histo->GetNbinsX(),
					     histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

  TH1D * sigmaMassVsEtaTH1D = new TH1D(type, type, histo->GetNbinsX(),
				       histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  TH1D * sigmaMassVsEtaPlusErrTH1D = new TH1D(type+"PlusErr", type+"PlusErr", histo->GetNbinsX(),
					      histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  TH1D * sigmaMassVsEtaMinusErrTH1D = new TH1D(type+"MinusErr", type+"MinusErr", histo->GetNbinsX(),
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
    // std::cout << "filling " << i << std::endl;
    posErrors[i-1] = valuesPlus[i] - values[i];
    if( valuesMinus[i] < 0 ) negErrors[i-1] = values[i];
    else negErrors[i-1] = values[i] - valuesMinus[i];

    graphAsymmErrors->SetPointEYlow(i-1, negErrors[i-1]);
    graphAsymmErrors->SetPointEYhigh(i-1, posErrors[i-1]);
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

  // Mass
  sigmaMassVsEtaPlusErrTH1D->Write();
  sigmaMassVsEtaTH1D->Write();
  sigmaMassVsEtaMinusErrTH1D->Write();

  sigmaMassVsEtaPlusErr_->Write();
  sigmaMassVsEta_->Write();
  sigmaMassVsEtaMinusErr_->Write();

  canvas->Write();
}

// ------------ method called to for each event  ------------
void ErrorsPropagationAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

void ErrorsPropagationAnalyzer::fillHistograms()
{
  std::cout << "Reading muon pairs from Root Tree in " << treeFileName_ << std::endl;
  RootTreeHandler rootTreeHandler;

  typedef std::vector<std::pair<lorentzVector,lorentzVector> > MuonPairVector;
  MuonPairVector savedPair;
  rootTreeHandler.readTree(maxEvents_, treeFileName_, &savedPair, 0);
  // rootTreeHandler.readTree(maxEvents, inputRootTreeFileName_, &savedPair, &(MuScleFitUtils::genPair));

  resolutionFunctionBase<std::vector<double> > * resolutionFunctionForVec = resolutionFunctionVecService( resolFitType_ );
  MuScleFitUtils::resolutionFunction = resolutionFunctionService( resolFitType_ );
  MuScleFitUtils::debugMassResol_ = false;
  MuScleFitUtils::debug = 0;
  MuScleFitUtils::resfind = std::vector<int>(6, 0);

  SigmaPt sigmaPt( parameters_, errors_ );
  SigmaPtDiff sigmaPtDiff;

  // Loop on all the pairs
  unsigned int i = 0;
  MuonPairVector::iterator it = savedPair.begin();
  std::cout << "Starting loop on " << savedPair.size() << " muons" << std::endl;
  for( ; it != savedPair.end(); ++it, ++i ) {
    double pt1 = it->first.pt();
    double eta1 = it->first.eta();
    double pt2 = it->second.pt();
    double eta2 = it->second.eta();

    if( debug_ ) {
      std::cout << "pt1 = " << pt1 << ", eta1 = " << eta1 << ", pt2 = " << pt2 << ", eta2 = " << eta2 << std::endl;
    }
    // double fabsEta1 = fabs(eta1);
    // double fabsEta2 = fabs(eta2);

    if( pt1 == 0 && pt2 == 0 && eta1 == 0 && eta2 == 0 ) continue;


    // double sigmaPt1 = sigmaPt( eta1 );
    // double sigmaPt2 = sigmaPt( eta2 );

    // double sigmaPtPlusErr1 = sigmaPt1 + sigmaPt.sigma(eta1);
    // double sigmaPtPlusErr2 = sigmaPt2 + sigmaPt.sigma(eta2);
    // double sigmaPtMinusErr1 = sigmaPt1 - sigmaPt.sigma(eta1);
    // double sigmaPtMinusErr2 = sigmaPt2 - sigmaPt.sigma(eta2);

    double sigmaPt1 = resolutionFunctionForVec->sigmaPt( pt1,eta1,parameters_ );
    double sigmaPt2 = resolutionFunctionForVec->sigmaPt( pt2,eta2,parameters_ );
    double sigmaPtPlusErr1 = sigmaPt1 + resolutionFunctionForVec->sigmaPtError( pt1,eta1,parameters_,errors_);
    double sigmaPtPlusErr2 = sigmaPt2 + resolutionFunctionForVec->sigmaPtError( pt2,eta2,parameters_,errors_ );
    double sigmaPtMinusErr1 = sigmaPt1 - resolutionFunctionForVec->sigmaPtError( pt1,eta1,parameters_,errors_ );
    double sigmaPtMinusErr2 = sigmaPt2 - resolutionFunctionForVec->sigmaPtError( pt2,eta2,parameters_,errors_ );

    double sigmaMass = MuScleFitUtils::massResolution( it->first, it->second, parameters_ );
    double sigmaMassPlusErr = MuScleFitUtils::massResolution( it->first, it->second, valuePlusError_ );
    double sigmaMassMinusErr = MuScleFitUtils::massResolution( it->first, it->second, valueMinusError_ );

    if( debug_ ) {
      std::cout << "sigmaPt1 = " << sigmaPt1 << " + " << sigmaPtPlusErr1 << " - " << sigmaPtMinusErr1 << std::endl;
      std::cout << "sigmaPt2 = " << sigmaPt2 << " + " << sigmaPtPlusErr2 << " - " << sigmaPtMinusErr2 << std::endl;
      std::cout << "sigmaMass = " << sigmaMass << " + " << sigmaMassPlusErr << " - " << sigmaMassMinusErr << std::endl;
    }

    // Protections from nans
    if( pt1 != pt1 ) continue;
    if( pt2 != pt2 ) continue;
    if( eta1 != eta1 ) continue;
    if( eta2 != eta2 ) continue;
    if( sigmaPt1 != sigmaPt1 ) continue;
    if( sigmaPt2 != sigmaPt2 ) continue;
    if( sigmaPtPlusErr1 != sigmaPtPlusErr1 ) continue;
    if( sigmaPtPlusErr2 != sigmaPtPlusErr2 ) continue;
    if( sigmaPtMinusErr1 != sigmaPtMinusErr1 ) continue;
    if( sigmaPtMinusErr2 != sigmaPtMinusErr2 ) continue;
    if( sigmaMass != sigmaMass ) continue;
    if( sigmaMassPlusErr != sigmaMassPlusErr ) continue;
    if( sigmaMassMinusErr != sigmaMassMinusErr ) continue;

    std::cout << "Muon pair number " << i << std::endl;

    // std::cout << "sigmaPt1 = " << sigmaPt1 << ", sigmaPt2 = " << sigmaPt2 << std::endl;
    // std::cout << "sigmaPtPlusErr1 = " << sigmaPtPlusErr1 << ", sigmaPtMinusErr2 = " << sigmaPtMinusErr2 << std::endl;
    // std::cout << "sigmaPtMinusErr1 = " << sigmaPtPlusErr1 << ", sigmaPtMinusErr2 = " << sigmaPtMinusErr2 << std::endl;


    if( (pt1 >= ptMinCut_ && pt1 <= ptMaxCut_) && (fabs(eta1) >= etaMinCut_ && fabs(eta1) <= etaMaxCut_) ) {
      sigmaPtVsPt_->Fill(pt1, sigmaPt1);
      sigmaPtVsPtPlusErr_->Fill(pt1, sigmaPtPlusErr1);
      sigmaPtVsPtMinusErr_->Fill(pt1, sigmaPtMinusErr1);
      sigmaPtVsPtDiff_->Fill(pt1, sigmaPtDiff.squaredDiff(eta1));
      sigmaMassVsPt_->Fill(pt1, sigmaMass);
      sigmaMassVsPtPlusErr_->Fill(pt1, sigmaMassPlusErr);
      sigmaMassVsPtMinusErr_->Fill(pt1, sigmaMassMinusErr);

      sigmaPtVsEta_->Fill(eta1, sigmaPt1);
      sigmaPtVsEtaPlusErr_->Fill(eta1, sigmaPtPlusErr1);
      sigmaPtVsEtaMinusErr_->Fill(eta1, sigmaPtMinusErr1);
      sigmaPtVsEtaDiff_->Fill(eta1, sigmaPtDiff.squaredDiff(eta1));
      sigmaMassVsEta_->Fill(eta1, sigmaMass);
      sigmaMassVsEtaPlusErr_->Fill(eta1, sigmaMassPlusErr);
      sigmaMassVsEtaMinusErr_->Fill(eta1, sigmaMassMinusErr);
    }
    if( (pt2 >= ptMinCut_ && pt2 <= ptMaxCut_) && (fabs(eta2) >= etaMinCut_ && fabs(eta2) <= etaMaxCut_) ) {
      sigmaPtVsPt_->Fill(pt2, sigmaPt2);
      sigmaPtVsPtPlusErr_->Fill(pt2, sigmaPtPlusErr2);
      sigmaPtVsPtMinusErr_->Fill(pt2, sigmaPtMinusErr2);
      sigmaPtVsPtDiff_->Fill(pt2, sigmaPtDiff.squaredDiff(eta2));
      sigmaMassVsPt_->Fill(pt2, sigmaMass);
      sigmaMassVsPtPlusErr_->Fill(pt2, sigmaMassPlusErr);
      sigmaMassVsPtMinusErr_->Fill(pt2, sigmaMassMinusErr);

      sigmaPtVsEta_->Fill(eta2, sigmaPt2);
      sigmaPtVsEtaPlusErr_->Fill(eta2, sigmaPtPlusErr2);
      sigmaPtVsEtaMinusErr_->Fill(eta2, sigmaPtMinusErr2);
      sigmaPtVsEtaDiff_->Fill(eta2, sigmaPtDiff.squaredDiff(eta2));
      sigmaMassVsEta_->Fill(eta2, sigmaMass);
      sigmaMassVsEtaPlusErr_->Fill(eta2, sigmaMassPlusErr);
      sigmaMassVsEtaMinusErr_->Fill(eta2, sigmaMassMinusErr);
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ErrorsPropagationAnalyzer);

#endif
