#include "DQM/SiStripCommissioningAnalysis/interface/PedsFullNoiseAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/PedsFullNoiseAnalysis.h" 
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
PedsFullNoiseAlgorithm::PedsFullNoiseAlgorithm( const edm::ParameterSet & pset, PedsFullNoiseAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    hPeds_(0,""),
    hNoise_(0,""),
    hNoise1D_(0,""),
    deadStripMax_(pset.getParameter<double>("DeadStripMax")),
    noisyStripMin_(pset.getParameter<double>("NoisyStripMin")),
    noiseDef_(pset.getParameter<std::string>("NoiseDefinition")),
    ksProbCut_(pset.getParameter<double>("KsProbCut"))
{
  //LogDebug(mlCommissioning_)
  //  << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
  //  << " Set maximum noise deviation for dead strip determination to: " << deadStripMax_;
	// LogDebug(mlCommissioning_)
  //  << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
  //  << " Set minimal noise deviation for noise strip determination to: " << noisyStripMin_;
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

  // Check number of histograms
  if ( histos.size() != 3 ) { 
    anal()->addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) { 
    anal()->fedKey( extractFedKey( histos.front() ) );
  }
  
  // Extract 1D histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }
    
// TO BE UPDATED!!!
    // Check run type
    SiStripHistoTitle title( (*ihis)->GetName() );
/*    if ( title.runType() != sistrip::PEDS_FULL_NOISE ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }
*/
    // Extract peds histos
    if ( title.extraInfo().find(sistrip::extrainfo::roughPedestals_) != std::string::npos ) {
      //@@ something here for rough peds?
    } else if ( title.extraInfo().find(sistrip::extrainfo::pedestals_) != std::string::npos ) {
      hPeds_.first = *ihis;
      hPeds_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::extrainfo::commonMode_) != std::string::npos ) {
      //@@ something here for CM plots?
    } else if ( title.extraInfo().find(sistrip::extrainfo::noiseProfile_) != std::string::npos ) {
      //@@ something here for noise profile plot?
      hNoise1D_.first = *ihis;
      hNoise1D_.second = (*ihis)->GetName();
    } else if ( title.extraInfo().find(sistrip::extrainfo::noise2D_) != std::string::npos ) {
      hNoise_.first = *ihis;
      hNoise_.second = (*ihis)->GetName();
    } else { 
      anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
    }  
  }

}

// -----------------------------------------------------------------------------
// 
void PedsFullNoiseAlgorithm::analyse() {

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }

  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  PedsFullNoiseAnalysis* ana = dynamic_cast<PedsFullNoiseAnalysis*>( tmp );
  if ( !ana ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

  if ( !hPeds_.first ) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !hNoise_.first ) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  TProfile * peds_histo = dynamic_cast<TProfile *>(hPeds_.first);
  TH2S * noise_histo 		= dynamic_cast<TH2S *>(hNoise_.first);
  if ( !peds_histo ) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !noise_histo ) {
    ana->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( peds_histo->GetNbinsX() != 256 ) {
    ana->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  if ( noise_histo->GetNbinsY() != 256 ) { // X range is configurable
    ana->addErrorCode(sistrip::numberOfBins_);
    return;
  }
  
	// Iterate through APVs 
	for ( uint16_t iapv = 0; iapv < 2; iapv++ ) {
  	
    // Used to calc mean and rms for peds and noise
    float p_sum = 0., p_sum2 = 0., p_max = -1.*sistrip::invalid_, p_min = sistrip::invalid_;
    float n_sum = 0., n_sum2 = 0., n_max = -1.*sistrip::invalid_, n_min = sistrip::invalid_;
    float r_sum = 0., r_sum2 = 0., r_max = -1.*sistrip::invalid_, r_min = sistrip::invalid_;		   
    // Iterate through strips of APV
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
    	
      ana->ksProb_[iapv].push_back(0);
      ana->noiseGaus_[iapv].push_back(0);
      ana->noiseBin84_[iapv].push_back(0);
      ana->noiseRMS_[iapv].push_back(0);
      ana->noiseSignif_[iapv].push_back(0);
    
      // pedestals and raw noise
      if ( peds_histo ) {
	    	if ( peds_histo->GetBinEntries(iapv*128 + istr + 1) ) {
	    		ana->peds_[iapv][istr] = peds_histo->GetBinContent(iapv*128 + istr + 1);
	    		p_sum += ana->peds_[iapv][istr];
	    		p_sum2 += (ana->peds_[iapv][istr] * ana->peds_[iapv][istr]);
	    		if ( ana->peds_[iapv][istr] > p_max ) { p_max = ana->peds_[iapv][istr];}
	    		if ( ana->peds_[iapv][istr] < p_min ) { p_min = ana->peds_[iapv][istr];}
	    		ana->raw_[iapv][istr] = peds_histo->GetBinError(iapv*128 + istr + 1);
	    		r_sum += ana->raw_[iapv][istr];
	    		r_sum2 += (ana->raw_[iapv][istr] * ana->raw_[iapv][istr]);
	    		if ( ana->raw_[iapv][istr] > r_max ) { r_max = ana->raw_[iapv][istr]; }
	    		if ( ana->raw_[iapv][istr] < r_min ) { r_min = ana->raw_[iapv][istr]; }
	    	}
      }
      // Noise from Full Distribution
      if ( noise_histo ) {
      	// Fit the ADC Distribution from TH2S by projecting it out and fitting.
        
        TH1S * noisehist = new TH1S("noisehist","",noise_histo->GetNbinsX(),
        						-noise_histo->GetNbinsX()/2,noise_histo->GetNbinsX()/2);
                    
        
        for(int i=0;i<=noise_histo->GetNbinsX()+1;i++){
         noisehist->SetBinContent(i,noise_histo->GetBinContent(i,iapv*128 + istr + 1));
        }           
        // If the histogram is valid.
        if(noisehist->Integral() > 0){
        	ana->noiseRMS_[iapv][istr] = noisehist->GetRMS();
        	noisehist->Fit("gaus","Q");                       
        	ana->noiseGaus_[iapv][istr] 	= noisehist->GetFunction("gaus")->GetParameter(2);
          
          // new Bin84 method
      		std::vector<float> integralFrac;
	    		integralFrac.push_back(1.*noisehist->GetBinContent(0)/noisehist->Integral(0,noisehist->GetNbinsX()));
      		// Calculate the integral of distro as a function of bins.
      		for(int i = 1; i < noisehist->GetNbinsX();i++){
	    		integralFrac.push_back(float(noisehist->GetBinContent(i))/
                noisehist->Integral(0,noisehist->GetNbinsX())+integralFrac[i-1]);
      			//Take the two bins next to 84% and solve for x in 0.84 = mx+b
      			if (integralFrac[i] >= 0.84135 && integralFrac[i-1] < 0.84135) {
      			// my quadratic noise calculation
      				double w = noisehist->GetBinWidth(i);
      				double a = noisehist->GetBinContent(i-1);
      				double b = noisehist->GetBinContent(i);
      				double f = w*(0.84135 -integralFrac[i-1])/(integralFrac[i]-integralFrac[i-1]);
      				double x = 0;
      				if (a==b) {
       					x = f;
      				} else {
       					double aa = (b-a)/(2*w);
       					double bb = (b+a)/2;
       					double cc = -b*f;
       					double dd = bb*bb-4*aa*cc; //if (dd<0) dd=0;
       					x = (-bb+sqrt(dd))/(2*aa);
      				}
      				ana->noiseBin84_[iapv][istr] = noisehist->GetBinLowEdge(i) + x;
      			}
          } 
        	// Compare shape of ADC to a histogram made of Gaus Fit for KSProb, Chi2Prob, Etc...
        	TH1D * FitHisto = new TH1D("FitHisto","FitHisto",noisehist->GetNbinsX(),
                           -noisehist->GetNbinsX()/2,noisehist->GetNbinsX()/2);
        	FitHisto->Add(noisehist->GetFunction("gaus"));
        	FitHisto->Sumw2();
        	noisehist->Sumw2();
          
        	if(FitHisto->Integral() > 0){
        		// This is stored as a float but will be plotted as an int at the summary histos.
            // This forces the histo to draw 10000 bins instead of 1.
        		ana->ksProb_[iapv][istr] = noisehist->KolmogorovTest(FitHisto)*10000;
        	}            	
        	delete FitHisto;
        }
        delete noisehist;    
      }    
      // Assigning the actual noise values used for Upload!!!!!!!!!!!!!!!!!!!!
      if (noiseDef_ == "Bin84") {
        if (ana->noiseBin84_[iapv][istr] > 0) {
          ana->noise_[iapv][istr] = ana->noiseBin84_[iapv][istr];
        } else {
          ana->noise_[iapv][istr] = ana->noiseRMS_[iapv][istr];
        }
      } else if (noiseDef_ == "RMS") {
        ana->noise_[iapv][istr] = ana->noiseRMS_[iapv][istr];
      } else edm::LogWarning(mlCommissioning_)<< "[PedsFullNoiseAlgorithm::"
      				<< __func__ << "]"<< " Unknown noise definition!!!";
      
      // Setting Sum of RMS and RMS^2 for Dead/Noisy Strip calculations
      n_sum += ana->noise_[iapv][istr];
      n_sum2 += (ana->noise_[iapv][istr] * ana->noise_[iapv][istr]);
      if ( ana->noise_[iapv][istr] > n_max ) { n_max = ana->noise_[iapv][istr]; }
      if ( ana->noise_[iapv][istr] < n_min ) { n_min = ana->noise_[iapv][istr]; }
      
    } // strip loop
  
    // Calc mean and rms for peds
    if ( !ana->peds_[iapv].empty() ) { 
    	p_sum /= static_cast<float>( ana->peds_[iapv].size() );
      p_sum2 /= static_cast<float>( ana->peds_[iapv].size() );
      ana->pedsMean_[iapv] = p_sum;
      ana->pedsSpread_[iapv] = sqrt( fabs(p_sum2 - p_sum*p_sum) );
    }
  
    // Calc mean and rms for noise using noiseRMS.
    if ( !ana->noise_[iapv].empty() ) { 
    	n_sum /= static_cast<float>( ana->noise_[iapv].size() );
      n_sum2 /= static_cast<float>( ana->noise_[iapv].size() );
      ana->noiseMean_[iapv] = n_sum;
      ana->noiseSpread_[iapv] = sqrt( fabs(n_sum2 - n_sum*n_sum) );
    }

    // Calc mean and rms for raw noise
    if ( !ana->raw_[iapv].empty() ) { 
    	r_sum /= static_cast<float>( ana->raw_[iapv].size() );
      r_sum2 /= static_cast<float>( ana->raw_[iapv].size() );
      ana->rawMean_[iapv] = r_sum;
      ana->rawSpread_[iapv] = sqrt( fabs(r_sum2 - r_sum*r_sum) );
    }
      
    // Set max and min values for peds, noise and raw noise
    if ( p_max > -1.*sistrip::maximum_ ) { ana->pedsMax_[iapv] = p_max; }
    if ( p_min < 1.*sistrip::maximum_ )  { ana->pedsMin_[iapv] = p_min; }
    if ( n_max > -1.*sistrip::maximum_ ) { ana->noiseMax_[iapv] = n_max; }
    if ( n_min < 1.*sistrip::maximum_ )  { ana->noiseMin_[iapv] = n_min; }
    if ( r_max > -1.*sistrip::maximum_ ) { ana->rawMax_[iapv] = r_max; }
    if ( r_min < 1.*sistrip::maximum_ )  { ana->rawMin_[iapv] = r_min; }
  
    // Set dead and noisy strips
    for ( uint16_t istr = 0; istr < 128; istr++ ) { // strip loop 
      // Set the significance of the noise of each strip also compared to apv avg.
      ana->noiseSignif_[iapv][istr] = (ana->noise_[iapv][istr]-ana->noiseMean_[iapv])/ana->noiseSpread_[iapv];
      if ( ana->noiseMin_[iapv] > sistrip::maximum_ || ana->noiseMax_[iapv] > sistrip::maximum_ ) { 
      	continue; 
      }
      // Strip Masking for Dead Strips
      if(ana->noiseSignif_[iapv][istr] < -deadStripMax_){
      	ana->dead_[iapv].push_back(istr);
        SiStripFecKey fec_key(ana->fecKey());
        LogTrace(mlDqmClient_)<<"DeadSignif "<<ana->noiseSignif_[iapv][istr]
        <<" "<<fec_key.fecCrate()
        <<" "<<fec_key.fecSlot()
        <<" "<<fec_key.fecRing()
        <<" "<<fec_key.ccuAddr()
        <<" "<<fec_key.ccuChan()
        <<" "<<fec_key.lldChan()
        <<" "<<iapv*128+istr<<std::endl;
      } // Strip Masking for Dead Strips
      // Laurent's Method for Flagging bad strips in TIB
      else if((ana->noiseMax_[iapv]/ana->noiseMean_[iapv] > 3 || ana->noiseSpread_[iapv] > 3)
      && ana->noiseSignif_[iapv][istr] > 1){
      	ana->noisy_[iapv].push_back(istr);
        SiStripFecKey fec_key(ana->fecKey());
        LogTrace(mlDqmClient_)<<"NoisyLM "<<ana->noiseMax_[iapv]/ana->noiseMean_[iapv]     
        <<" "<<fec_key.fecCrate()
        <<" "<<fec_key.fecSlot()
        <<" "<<fec_key.fecRing()
        <<" "<<fec_key.ccuAddr()
        <<" "<<fec_key.ccuChan()
        <<" "<<fec_key.lldChan()
        <<" "<<iapv*128+istr<<std::endl;
      } // if NoisyLM 
      //Strip Masking for Non Gassian Strips which are also noisy.
      else if(ana->ksProb_[iapv][istr] < ksProbCut_){
      	ana->noisy_[iapv].push_back(istr);
        SiStripFecKey fec_key(ana->fecKey());  
        LogTrace(mlDqmClient_)<<"NoisyKS "<<ana->ksProb_[iapv][istr] 
        <<" "<<fec_key.fecCrate()
        <<" "<<fec_key.fecSlot()
        <<" "<<fec_key.fecRing()
        <<" "<<fec_key.ccuAddr()
        <<" "<<fec_key.ccuChan()
        <<" "<<fec_key.lldChan()
        <<" "<<iapv*128+istr<<std::endl;
      } //Strip Masking for Non Gassian Strips which are also noisy.
      else if(ana->noiseSignif_[iapv][istr] > noisyStripMin_){
      	ana->noisy_[iapv].push_back(istr);
        SiStripFecKey fec_key(ana->fecKey());
        LogTrace(mlDqmClient_)<<"NoisySignif "<<ana->noiseSignif_[iapv][istr]    
        <<" "<<fec_key.fecCrate()
        <<" "<<fec_key.fecSlot()
        <<" "<<fec_key.fecRing()
        <<" "<<fec_key.ccuAddr()
        <<" "<<fec_key.ccuChan()
        <<" "<<fec_key.lldChan()
        <<" "<<iapv*128+istr<<std::endl;
      } // if Signif 
    }// strip loop to set dead or noisy strips   
	} // apv loop
  //std::cout << std::endl;
}
