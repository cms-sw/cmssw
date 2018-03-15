#include "DQM/SiStripCommissioningAnalysis/interface/PedsFullNoiseAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/PedsFullNoiseAnalysis.h" 
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TFitResult.h"
#include "TMath.h"
#include "Math/DistFunc.h"
#include "Math/ProbFuncMathCore.h"
#include "Fit/BinData.h"
#include "HFitInterface.h"
#include "Math/GoFTest.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;
using namespace std;
// ----------------------------------------------------------------------------
// 
PedsFullNoiseAlgorithm::PedsFullNoiseAlgorithm( const edm::ParameterSet & pset, PedsFullNoiseAnalysis* const anal ):
  CommissioningAlgorithm(anal),
  hPeds_(nullptr,""),
  hNoise_(nullptr,""),
  hNoise2D_(nullptr,""),
  maxDriftResidualCut_(pset.getParameter<double>("MaxDriftResidualCut")),
  minStripNoiseCut_(pset.getParameter<double>("MinStripNoiseCut")),
  maxStripNoiseCut_(pset.getParameter<double>("MaxStripNoiseCut")),
  maxStripNoiseSignificanceCut_(pset.getParameter<double>("MaxStripNoiseSignificanceCut")),
  adProbabCut_(pset.getParameter<double>("AdProbabCut")),
  ksProbabCut_(pset.getParameter<double>("KsProbabCut")),
  generateRandomHisto_(pset.getParameter<bool>("GenerateRandomHisto")),
  jbProbabCut_(pset.getParameter<double>("JbProbabCut")),
  chi2ProbabCut_(pset.getParameter<double>("Chi2ProbabCut")),
  kurtosisCut_(pset.getParameter<double>("KurtosisCut")),
  integralTailCut_(pset.getParameter<double>("IntegralTailCut")),
  integralNsigma_(pset.getParameter<int>("IntegralNsigma")),
  ashmanDistance_(pset.getParameter<double>("AshmanDistance")),
  amplitudeRatio_(pset.getParameter<double>("AmplitudeRatio"))
{
  LogDebug(mlCommissioning_)
    << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
    << " Set maximum drift of the mean value to: " << maxDriftResidualCut_ 
    << " Set minimum noise value to: " << minStripNoiseCut_
    << " Set maximum noise value to: " << maxStripNoiseCut_
    << " Set maximum noise significance value to: " << maxStripNoiseSignificanceCut_
    << " Set minimum Anderson-Darling p-value to: " << adProbabCut_
    << " Set minimum Kolmogorov-Smirnov p-value to: " << ksProbabCut_
    << " Set minimum Jacque-Bera p-value to: " << jbProbabCut_
    << " Set minimum Chi2 p-value to: " << chi2ProbabCut_
    << " Set N-sigma for the integral to : " << integralNsigma_
    << " Set maximum integral tail at N-sigma to : " << integralTailCut_
    << " Set maximum Kurtosis to : " << kurtosisCut_;
}
    
// ----------------------------------------------------------------------------
//
 
void PedsFullNoiseAlgorithm::extract( const std::vector<TH1*>& histos ) { 
  
  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
      << " NULL pointer to Analysis object!";
    return; 
  }

  // Check number of histograms --> Pedestal, noise and noise2D
  if ( histos.size() != 3 ) { 
    anal()->addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title --> i.e. APV pairs or LLD channel
  if ( !histos.empty() ) { 
    anal()->fedKey( extractFedKey( histos.front() ) );
  }
  
  // Extract 1D histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }
    
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::PEDS_FULL_NOISE ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
    
    // Extract peds histos
    if ( title.extraInfo().find(sistrip::extrainfo::roughPedestals_) != std::string::npos ) {
      //@@ something here for rough peds?
    } 
    else if ( title.extraInfo().find(sistrip::extrainfo::pedestals_) != std::string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
    } 
    else if ( title.extraInfo().find(sistrip::extrainfo::commonMode_) != std::string::npos ) {
      //@@ something here for CM plots?
    } 
    else if ( title.extraInfo().find(sistrip::extrainfo::noiseProfile_) != std::string::npos ) {
      //@@ something here for noise profile plot?
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
    } 
    else if ( title.extraInfo().find(sistrip::extrainfo::noise2D_) != std::string::npos ) {
      hNoise2D_.first = *ihis;
      hNoise2D_.second = (*ihis)->GetName();
    } 
    else { 
      anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
    }  
  }  
}

// resetting vectors
void PedsFullNoiseAlgorithm::reset(PedsFullNoiseAnalysis* ana){

  for(size_t iapv = 0 ; iapv < ana->peds_.size(); iapv++){
    ana->pedsMean_[iapv] = 0.; 
    ana->rawMean_[iapv]  = 0.; 
    ana->noiseMean_[iapv] = 0.;
    ana->pedsSpread_[iapv] = 0.;
    ana->noiseSpread_[iapv] = 0.;
    ana->rawSpread_[iapv] = 0.;
    ana->pedsMax_[iapv] = 0.;
    ana->pedsMin_[iapv] = 0.;
    ana->rawMax_[iapv] = 0.;
    ana->rawMin_[iapv] = 0.;
    ana->noiseMax_[iapv] = 0.;
    ana->noiseMin_[iapv] = 0.;

    for(size_t istrip = 0; istrip < ana->peds_[iapv].size(); istrip++){
      ana->peds_[iapv][istrip]  = 0.;
      ana->noise_[iapv][istrip] = 0.;
      ana->raw_[iapv][istrip]   = 0.;
      ana->adProbab_[iapv][istrip] = 0.;
      ana->ksProbab_[iapv][istrip] = 0.;
      ana->jbProbab_[iapv][istrip] = 0.;
      ana->chi2Probab_[iapv][istrip] = 0.;
      ana->residualRMS_[iapv][istrip] = 0.;
      ana->residualSigmaGaus_[iapv][istrip] = 0.;
      ana->noiseSignificance_[iapv][istrip] = 0.;
      ana->residualMean_[iapv][istrip] = 0.;
      ana->residualSkewness_[iapv][istrip] = 0.;
      ana->residualKurtosis_[iapv][istrip] = 0.;
      ana->residualIntegralNsigma_[iapv][istrip] = 0.;
      ana->residualIntegral_[iapv][istrip] = 0.;      
      ana->deadStripBit_[iapv][istrip] = 0;
      ana->badStripBit_[iapv][istrip] = 0;
    }
  }
}

// -----------------------------------------------------------------------------
// 
void PedsFullNoiseAlgorithm::analyse() {

  // check base analysis object
  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }
  
  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );  
  PedsFullNoiseAnalysis* ana = dynamic_cast<PedsFullNoiseAnalysis*>( tmp );

  // check PedsFullNoiseAnalysis object
  if ( !ana ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }


  // check if the histograms exists 
  if ( !hPeds_.first) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  if ( !hNoise_.first ) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !hNoise2D_.first ) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  // take the histograms
  TProfile *histoPeds = dynamic_cast<TProfile *>(hPeds_.first);
  TProfile *histoNoiseMean = dynamic_cast<TProfile *>(hNoise_.first);
  TH2S * histoNoise = dynamic_cast<TH2S*>(hNoise2D_.first);

  // Make sanity checks about pointers
  if (not histoPeds) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if (not histoNoiseMean) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if (not histoNoise) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  // check the binning  --> each x-axis bin is 1 strip -> 2APV per lldChannel -> 256 strips
  if (histoPeds->GetNbinsX() != 256 ) {
    ana->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  //check the binning  --> each x-axis bin is 1 strip -> 2APV per lldChannel -> 256 strips
  if (histoNoiseMean->GetNbinsX() != 256 ) { 
    ana->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  //check the binning  --> each y-axis bin is 1 strip -> 2APV per lldChannel -> 256 strips
  if (histoNoise->GetNbinsY() != 256 ) { 
    ana->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  //Reset values
  reset(ana);
  

  // loop on each strip
  uint32_t apvID = -1;

  // Save basic information at strip / APV level
  vector<float> ped_max;
  vector<float> ped_min;
  vector<float> raw_max;
  vector<float> raw_min;
  vector<float> noise_max;
  vector<float> noise_min;

  // loop on each strip in the lldChannel
  for(int iStrip = 0; iStrip < histoPeds->GetNbinsX(); iStrip++){

    if(iStrip < histoPeds->GetNbinsX()/2) apvID = 0;
    else apvID = 1;    
    
    int stripBin = 0;
    if(iStrip >= 128) stripBin = iStrip-128;
    else stripBin = iStrip;
      
    ana->peds_[apvID][stripBin]  = histoPeds->GetBinContent(iStrip+1); // pedestal value
    ana->noise_[apvID][stripBin] = histoNoiseMean->GetBinContent(iStrip+1); // noise value 
    ana->raw_[apvID][stripBin]   = histoPeds->GetBinError(iStrip+1); // raw noise value
 
    ana->pedsMean_[apvID]  += ana->peds_[apvID][stripBin];  // mean pedestal
    ana->rawMean_[apvID]   += ana->raw_[apvID][stripBin]; // mean raw noise
    ana->noiseMean_[apvID] += ana->noise_[apvID][stripBin];  // mean noise

    // max pedestal
    if(ped_max.size() < apvID+1)
      ped_max.push_back(ana->peds_[apvID][stripBin]);
    else{
      if(ana->peds_[apvID][stripBin] > ped_max.at(apvID))
	ped_max.at(apvID) = ana->peds_[apvID][stripBin]; 
    }

    // min pedestal
    if(ped_min.size() < apvID+1)
      ped_min.push_back(ana->peds_[apvID][stripBin]);
    else{
      if(ana->peds_[apvID][stripBin] < ped_min.at(apvID))
	ped_min.at(apvID) = ana->peds_[apvID][stripBin]; // min pedestal
    }

    // max noise
    if(noise_max.size() < apvID+1)
      noise_max.push_back(ana->noise_[apvID][stripBin]);
    else{
      if(ana->noise_[apvID][stripBin] > noise_max.at(apvID))
	noise_max.at(apvID) = ana->noise_[apvID][stripBin]; 
    }

    // min noise
    if(noise_min.size() < apvID+1)
      noise_min.push_back(ana->noise_[apvID][stripBin]);
    else{
      if(ana->noise_[apvID][stripBin] < noise_min.at(apvID))
	noise_min.at(apvID) = ana->noise_[apvID][stripBin]; 
    }

    // max raw
    if(raw_max.size() < apvID+1)
      raw_max.push_back(ana->raw_[apvID][stripBin]);
    else{
      if(ana->raw_[apvID][stripBin] > raw_max.at(apvID))
	raw_max.at(apvID) = ana->raw_[apvID][stripBin]; 
    }

    // min raw
    if(raw_min.size() < apvID+1)
      raw_min.push_back(ana->raw_[apvID][stripBin]);
    else{
      if(ana->raw_[apvID][stripBin] < raw_min.at(apvID))
	raw_min.at(apvID) = ana->raw_[apvID][stripBin]; 
    }          
  }

  // Mean values
  for(unsigned int iApv = 0; iApv < ana->pedsMean_.size(); iApv++){
    ana->pedsMean_.at(iApv) /= (ana->peds_[iApv].size()); // calculate mean pedestal per APV
    ana->rawMean_.at(iApv) /= (ana->raw_[iApv].size());  // calculate mean raw noise per APV
    ana->noiseMean_.at(iApv) /= (ana->noise_[iApv].size()); // calculate mean noise per APV
  }

  // Min and Max
  for(unsigned int iApv = 0; iApv < ped_max.size(); iApv++){
    if(ped_max.at(iApv) > sistrip::maximum_)
      ana->pedsMax_.at(iApv) = sistrip::maximum_;
    else if(ped_max.at(iApv) < -1.*sistrip::maximum_)
      ana->pedsMax_.at(iApv) = -1.*sistrip::maximum_;
    else
      ana->pedsMax_.at(iApv) = ped_max.at(iApv);

    if(ped_min.at(iApv) > sistrip::maximum_)
      ana->pedsMin_.at(iApv) = sistrip::maximum_;
    else if(ped_min.at(iApv) < -1.*sistrip::maximum_)
      ana->pedsMin_.at(iApv) = -1.*sistrip::maximum_;
    else
      ana->pedsMin_.at(iApv) = ped_min.at(iApv);

    if(noise_max.at(iApv) > sistrip::maximum_)
      ana->noiseMax_.at(iApv) = sistrip::maximum_;
    else if(noise_max.at(iApv) < -1.*sistrip::maximum_)
      ana->noiseMax_.at(iApv) = -1.*sistrip::maximum_;
    else
      ana->noiseMax_.at(iApv) = noise_max.at(iApv);

    if(noise_min.at(iApv) > sistrip::maximum_)
      ana->noiseMin_.at(iApv) = sistrip::maximum_;
    else if(noise_min.at(iApv) < -1.*sistrip::maximum_)
      ana->noiseMin_.at(iApv) = -1.*sistrip::maximum_;
    else
      ana->noiseMin_.at(iApv) = noise_min.at(iApv);

    if(raw_max.at(iApv) > sistrip::maximum_)
      ana->rawMax_.at(iApv) = sistrip::maximum_;
    else if(raw_max.at(iApv) < -1.*sistrip::maximum_)
      ana->rawMax_.at(iApv) = -1.*sistrip::maximum_;
    else
      ana->rawMax_.at(iApv) = raw_max.at(iApv);

    if(raw_min.at(iApv) > sistrip::maximum_)
      ana->rawMin_.at(iApv) = sistrip::maximum_;
    else if(raw_min.at(iApv) < -1.*sistrip::maximum_)
      ana->rawMin_.at(iApv) = -1.*sistrip::maximum_;
    else
      ana->rawMin_.at(iApv) = raw_min.at(iApv);

  }

  // Calculate the spread for noise and pedestal
  apvID = -1;

  for(int iStrip = 0; iStrip < histoNoiseMean->GetNbinsX(); iStrip++){
    if(iStrip < histoNoiseMean->GetNbinsX()/2) apvID = 0;
    else apvID = 1;
    ana->pedsSpread_[apvID]  += pow(histoPeds->GetBinContent(iStrip+1)-ana->pedsMean_.at(apvID),2);
    ana->noiseSpread_[apvID] += pow(histoNoiseMean->GetBinContent(iStrip+1)-ana->noiseMean_.at(apvID),2);
    ana->rawSpread_[apvID]   += pow(histoPeds->GetBinError(iStrip+1)-ana->rawMean_.at(apvID),2);    
  }

  for(unsigned int iApv = 0; iApv < ana->pedsSpread_.size(); iApv++){
    ana->pedsSpread_[iApv]  = sqrt(ana->pedsSpread_[iApv])/sqrt(ana->peds_[iApv].size() -1);
    ana->noiseSpread_[iApv] = sqrt(ana->noiseSpread_[iApv])/sqrt(ana->noise_[iApv].size()-1);
    ana->rawSpread_[iApv]   = sqrt(ana->rawSpread_[iApv])/sqrt(ana->raw_[iApv].size() -1);
  }

  // loop on each strip in the lldChannel
  apvID = 0;
  TH1S* histoResidualStrip = new TH1S("histoResidualStrip","",histoNoise->GetNbinsX(),histoNoise->GetXaxis()->GetXmin(),histoNoise->GetXaxis()->GetXmax());
  histoResidualStrip->Sumw2();
  histoResidualStrip->SetDirectory(nullptr);
  TF1* fitFunc   = new TF1 ("fitFunc","gaus(0)",histoNoise->GetXaxis()->GetXmin(),histoNoise->GetXaxis()->GetXmax());
  TF1* fit2Gaus  = nullptr;
  TH1F* randomHisto = nullptr;
  TFitResultPtr result;

  for(int iStrip = 0; iStrip < histoNoise->GetNbinsY(); iStrip++){
    // tell which APV
    if(iStrip < histoNoise->GetNbinsY()/2) apvID = 0;
    else apvID = 1;
    histoResidualStrip->Reset();

    int stripBin = 0;
    if(iStrip >= 128) stripBin = iStrip-128;
    else stripBin = iStrip;

    for(int iBinX = 0; iBinX < histoNoise->GetNbinsX(); iBinX++){
      histoResidualStrip->SetBinContent(iBinX+1,histoNoise->GetBinContent(iBinX+1,iStrip+1));
      histoResidualStrip->SetBinError(iBinX+1,histoNoise->GetBinError(iBinX+1,iStrip+1));    
    }
    
    if(histoResidualStrip->Integral() == 0){ // dead strip --> i.e. no data
      
      // set default values
      ana->adProbab_[apvID][stripBin] = 0;
      ana->ksProbab_[apvID][stripBin] = 0;
      ana->jbProbab_[apvID][stripBin] = 0;
      ana->chi2Probab_[apvID][stripBin] = 0; 
      ana->noiseSignificance_[apvID][stripBin] = 0;
      ana->residualMean_[apvID][stripBin] = 0;
      ana->residualRMS_[apvID][stripBin] = 0;
      ana->residualSigmaGaus_[apvID][stripBin] = 0;
      ana->residualSkewness_[apvID][stripBin] = 0;
      ana->residualKurtosis_[apvID][stripBin] = 0;
      ana->residualIntegralNsigma_[apvID][stripBin] = 0;
      ana->residualIntegral_[apvID][stripBin] = 0;
      ana->deadStrip_[apvID].push_back(stripBin);      
      ana->deadStripBit_[apvID][stripBin] = 1;
      ana->badStripBit_[apvID][stripBin] = 0;

      SiStripFecKey fec_key(ana->fecKey());
      LogTrace(mlDqmClient_)<<"DeadStrip: fecCrate "
			    <<" "<<fec_key.fecCrate()                                                                                                                           
			    <<" fecSlot "<<fec_key.fecSlot()                                                                                                                        
			    <<" fecRing "<<fec_key.fecRing()                                                                                                                               
			    <<" ccuAddr "<<fec_key.ccuAddr()                                                                                                                              
			    <<" ccChan  "<<fec_key.ccuChan()                                                                                                                                        
			    <<" lldChan "<<fec_key.lldChan()                                                                                                                                  
			    <<" apvID "<<apvID
			    <<" stripID "<<iStrip;
      

      continue;
    }
    
    // set / calculated basic quantities
    ana->residualMean_[apvID][stripBin]      = histoResidualStrip->GetMean();
    ana->residualRMS_[apvID][stripBin]       = histoResidualStrip->GetRMS();
    ana->residualSkewness_[apvID][stripBin]  = histoResidualStrip->GetSkewness();
    ana->residualKurtosis_[apvID][stripBin]  = histoResidualStrip->GetKurtosis();
    ana->noiseSignificance_[apvID][stripBin] = (ana->noise_[apvID][stripBin]-ana->noiseMean_[apvID])/ana->noiseSpread_[apvID];
    ana->residualIntegral_[apvID][stripBin]  = histoResidualStrip->Integral();
    ana->residualIntegralNsigma_[apvID][stripBin] = 
      (histoResidualStrip->Integral(histoResidualStrip->FindBin(ana->residualMean_[apvID][stripBin]+ana->residualRMS_[apvID][stripBin]*integralNsigma_),histoResidualStrip->GetNbinsX()+1) + 
       histoResidualStrip->Integral(0,histoResidualStrip->FindBin(ana->residualMean_[apvID][stripBin]-ana->residualRMS_[apvID][stripBin]*integralNsigma_)))/ana->residualIntegral_[apvID][stripBin];

    // performing a Gaussian fit of the residual distribution
    fitFunc->SetRange(histoNoise->GetXaxis()->GetXmin(),histoNoise->GetXaxis()->GetXmax());
    fitFunc->SetParameters(ana->residualIntegral_[apvID][stripBin],ana->residualMean_[apvID][stripBin],ana->residualRMS_[apvID][stripBin]);
    result = histoResidualStrip->Fit(fitFunc,"QSRN");

    // Good gaussian fit
    if(result.Get()){

      ana->residualSigmaGaus_[apvID][stripBin] = fitFunc->GetParameter(2);
      ana->chi2Probab_[apvID][stripBin]        = result->Prob();
      
      // jacque bera probability
      float jbVal = (ana->residualIntegral_[apvID][stripBin]/6)*(pow(ana->residualSkewness_[apvID][stripBin],2)+pow(ana->residualKurtosis_[apvID][stripBin],2)/4);
      ana->jbProbab_[apvID][stripBin] = ROOT::Math::chisquared_cdf_c(jbVal,2);
      
      //Kolmogorov Smirnov and Anderson Darlong
      if(randomHisto == nullptr)
	randomHisto = (TH1F*) histoResidualStrip->Clone("randomHisto");
      randomHisto->Reset();
      randomHisto->SetDirectory(nullptr);
      
      if(generateRandomHisto_){///
	randomHisto->FillRandom("fitFunc",histoResidualStrip->Integral());
	if(randomHisto->Integral() != 0){
	  ana->ksProbab_[apvID][stripBin] = histoResidualStrip->KolmogorovTest(randomHisto,"N");
	  ana->adProbab_[apvID][stripBin] = histoResidualStrip->AndersonDarlingTest(randomHisto);
	}
	else{
	  ana->ksProbab_[apvID][stripBin] = 0;
	  ana->adProbab_[apvID][stripBin] = 0;
	}
	    
      }
      else{
	randomHisto->Add(fitFunc);
	if(randomHisto->Integral() != 0){
	  ana->ksProbab_[apvID][stripBin] = histoResidualStrip->KolmogorovTest(randomHisto,"N");
	  ROOT::Fit::BinData data1;
	  ROOT::Fit::BinData data2;
	  ROOT::Fit::FillData(data1,histoResidualStrip,nullptr);
	  data2.Initialize(randomHisto->GetNbinsX()+1,1);
	  for(int ibin = 0; ibin < randomHisto->GetNbinsX(); ibin++){ 
	    if(histoResidualStrip->GetBinContent(ibin+1) != 0 or randomHisto->GetBinContent(ibin+1) >= 1)
	      data2.Add(randomHisto->GetBinCenter(ibin+1),randomHisto->GetBinContent(ibin+1),randomHisto->GetBinError(ibin+1));
	  }
	  
	  double probab, value;
	  ROOT::Math::GoFTest::AndersonDarling2SamplesTest(data1,data2,probab,value);
	  ana->adProbab_[apvID][stripBin] = probab;
	}
	else{
	  ana->ksProbab_[apvID][stripBin] = 0;
	  ana->adProbab_[apvID][stripBin] = 0;
	}
      }
    }
  
    // start applying selections storing output
    bool badStripFlag =  false;
    ana->deadStripBit_[apvID][stripBin] = 0; // is not dead if the strip has data

    if(fabs(ana->residualMean_[apvID][stripBin]) > maxDriftResidualCut_ and not badStripFlag) {//mean value
      ana->shiftedStrip_[apvID].push_back(stripBin);
      badStripFlag = true;
    }
    
    if(ana->residualRMS_[apvID][stripBin] < minStripNoiseCut_ and not badStripFlag){ // low noise
      ana->lowNoiseStrip_[apvID].push_back(stripBin);
      badStripFlag  = true;
    }

    if(ana->residualRMS_[apvID][stripBin] > maxStripNoiseCut_ and not badStripFlag){ // large noise
      ana->largeNoiseStrip_[apvID].push_back(stripBin);
      badStripFlag = true;
    }

    if(fabs(ana->noiseSignificance_[apvID][stripBin]) > maxStripNoiseSignificanceCut_ and not badStripFlag){ // large noise significance
      ana->largeNoiseSignificance_[apvID].push_back(stripBin);      
      badStripFlag = true;
    }
    
    if(result.Get() and result->Status() != 0) // bad fit status
      ana->badFitStatus_[apvID].push_back(stripBin);

    if(ana->adProbab_[apvID][stripBin] < adProbabCut_ and not badStripFlag) // bad AD p-value --> store the strip-id
      ana->badADProbab_[apvID].push_back(stripBin);

    if(ana->ksProbab_[apvID][stripBin] < ksProbabCut_ and not badStripFlag) // bad KS p-value --> store the strip-id 
      ana->badKSProbab_[apvID].push_back(stripBin);

    if(ana->jbProbab_[apvID][stripBin] < jbProbabCut_ and not badStripFlag) // bad JB p-value  --> store the strip-id 
      ana->badJBProbab_[apvID].push_back(stripBin);

    if(ana->chi2Probab_[apvID][stripBin] < chi2ProbabCut_ and not badStripFlag) // bad CHI2 p-value  --> store the strip-id 
      ana->badChi2Probab_[apvID].push_back(stripBin);

    if(ana->adProbab_[apvID][stripBin] < adProbabCut_ and ana->ksProbab_[apvID][stripBin] < ksProbabCut_ and   
       ana->jbProbab_[apvID][stripBin] < jbProbabCut_ and ana->chi2Probab_[apvID][stripBin] < chi2ProbabCut_)
      badStripFlag = true; // bad strip is flagged as bad by all the methods

    if(ana->residualKurtosis_[apvID][stripBin] > kurtosisCut_ and ana->residualIntegralNsigma_[apvID][stripBin] > integralTailCut_ and not badStripFlag){ // bad tails      
      ana->badTailStrip_[apvID].push_back(stripBin);
      badStripFlag = true;
    }

    if(badStripFlag){ // loop for double peaked
      
      fit2Gaus = new TF1("dgaus","[0]*exp(-((x-[1])*(x-[1]))/(2*[2]*[2]))+[3]*exp(-((x-[4])*(x-[4]))/(2*[5]*[5]))",histoNoise->GetXaxis()->GetXmin(),histoNoise->GetXaxis()->GetXmax());
      fit2Gaus->SetParameter(0,fitFunc->GetParameter(0)/2);
      fit2Gaus->SetParameter(3,fitFunc->GetParameter(0)/2);
      fit2Gaus->SetParameter(1,1.);
      fit2Gaus->SetParameter(4,-1.);
      fit2Gaus->SetParameter(2,fitFunc->GetParameter(2));
      fit2Gaus->SetParameter(5,fitFunc->GetParameter(2));
      fit2Gaus->SetParLimits(1,0.,histoNoise->GetXaxis()->GetXmax());
      fit2Gaus->SetParLimits(4,histoNoise->GetXaxis()->GetXmin(),0);
      result = histoResidualStrip->Fit(fit2Gaus,"QSR");

      // ashman distance
      float ashman = TMath::Power(2,0.5)*abs(fit2Gaus->GetParameter(1)-fit2Gaus->GetParameter(4))/(sqrt(pow(fit2Gaus->GetParameter(2),2)+pow(fit2Gaus->GetParameter(5),2))); 
      // amplitude
      float amplitudeRatio = std::min(fit2Gaus->GetParameter(0),fit2Gaus->GetParameter(3))/std::max(fit2Gaus->GetParameter(0),fit2Gaus->GetParameter(3));
      
      if(ashman > ashmanDistance_ and amplitudeRatio > amplitudeRatio_)
	ana->badDoublePeakStrip_[apvID].push_back(stripBin);      
    }
    
    if(badStripFlag){ // save the final bit

      ana->badStrip_[apvID].push_back(stripBin);
      ana->badStripBit_[apvID][stripBin] = 1;
      
      SiStripFecKey fec_key(ana->fecKey());
      LogTrace(mlDqmClient_)<<"BadStrip: fecCrate "
			    <<" "<<fec_key.fecCrate()                                                                                                                           
			    <<" fecSlot "<<fec_key.fecSlot()                                                                                                                        
			    <<" fecRing "<<fec_key.fecRing()                                                                                                                               
			    <<" ccuAddr "<<fec_key.ccuAddr()                                                                                                                              
			    <<" ccChan  "<<fec_key.ccuChan()                                                                                                                                        
			    <<" lldChan "<<fec_key.lldChan()                                                                                                                                  
			    <<" apvID "<<apvID
			    <<" stripID "<<stripBin;

    }
    else
      ana->badStripBit_[apvID][stripBin] = 0;    
  }

    
  ped_max.clear();
  ped_min.clear();
  raw_max.clear();
  raw_min.clear();
  noise_max.clear();
  noise_min.clear();
  if(histoResidualStrip) delete histoResidualStrip;
  if(fitFunc) delete fitFunc;
  if(randomHisto) delete randomHisto;
  if(fit2Gaus) delete fit2Gaus;
 
}
