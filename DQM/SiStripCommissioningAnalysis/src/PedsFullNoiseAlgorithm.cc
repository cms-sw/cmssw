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
    deadStripMax_(pset.getParameter<double>("DeadStripMax")),
    noisyStripMin_(pset.getParameter<double>("NoisyStripMin"))
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
  if ( histos.size() != 4 ) { 
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
  PedsFullNoiseAnalysis* anal = dynamic_cast<PedsFullNoiseAnalysis*>( tmp );
  if ( !anal ) {
    edm::LogWarning(mlCommissioning_)
      << "[PedsFullNoiseAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }

  if ( !hPeds_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !hNoise_.first ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  TProfile * peds_histo = dynamic_cast<TProfile *>(hPeds_.first);
  TH2S * noise_histo = dynamic_cast<TH2S *>(hNoise_.first);
  if ( !peds_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( !noise_histo ) {
    anal->addErrorCode(sistrip::nullPtr_);
    return;
  }

  if ( peds_histo->GetNbinsX() != 256 ) {
    anal->addErrorCode(sistrip::numberOfBins_);
    return;
  }

  if ( noise_histo->GetNbinsY() != 256 ) { // X range is configurable
    anal->addErrorCode(sistrip::numberOfBins_);
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

        anal->ksProb_[iapv].push_back(0);
        anal->chi2Prob_[iapv].push_back(0);
        anal->noiseGaus_[iapv].push_back(0);
        anal->bin84Percent_[iapv].push_back(0);
        anal->noiseSignif_[iapv].push_back(0);
        
        // pedestals and raw noise
        if ( peds_histo ) {
	    if ( peds_histo->GetBinEntries(iapv*128 + istr + 1) ) {
	        anal->peds_[iapv][istr] = peds_histo->GetBinContent(iapv*128 + istr + 1);
	        p_sum += anal->peds_[iapv][istr];
	        p_sum2 += (anal->peds_[iapv][istr] * anal->peds_[iapv][istr]);
	        if ( anal->peds_[iapv][istr] > p_max ) { p_max = anal->peds_[iapv][istr];}
	        if ( anal->peds_[iapv][istr] < p_min ) { p_min = anal->peds_[iapv][istr];}
	        anal->raw_[iapv][istr] = peds_histo->GetBinError(iapv*128 + istr + 1);
	        r_sum += anal->raw_[iapv][istr];
	        r_sum2 += (anal->raw_[iapv][istr] * anal->raw_[iapv][istr]);
	        if ( anal->raw_[iapv][istr] > r_max ) { r_max = anal->raw_[iapv][istr]; }
	        if ( anal->raw_[iapv][istr] < r_min ) { r_min = anal->raw_[iapv][istr]; }
	    }
        } 

        // Noise
        if ( noise_histo ) {
            TH1D * noisehist = noise_histo->ProjectionX("projx", iapv*128 + istr + 1, iapv*128 + istr + 1);
            // Gaussian Fit to set noise on each strip
            noisehist->Fit("gaus","Q");
            TF1 * gausFit = noisehist->GetFunction("gaus");
            anal->noiseGaus_[iapv][istr] = gausFit->GetParameter(2);
            TH1D * FitHisto = new TH1D("FitHisto","FitHisto",noisehist->GetNbinsX(),
                                       -noisehist->GetNbinsX()/2,noisehist->GetNbinsX()/2);
            FitHisto->Add(gausFit);
            FitHisto->Sumw2();
            noisehist->Sumw2();
            float KS = 0;
            if(noisehist->Integral() > 0){
            	KS = noisehist->KolmogorovTest(FitHisto);
                anal->ksProb_[iapv][istr] = KS*10000;
                anal->chi2Prob_[iapv][istr] = FitHisto->Chi2Test(noisehist, "OFUF")*10000;
            }
                      
            // Integral to 84% method to set noise of each strip.
            int current = 0;
            while(current < noisehist->GetNbinsX() && noisehist->Integral(0,current)/noisehist->Integral() < 0.842){
				current++;
			}
            anal->bin84Percent_[iapv][istr] = (current - noisehist->GetNbinsX()/2);            
            
            // Setting the noise of each strip
            anal->noise_[iapv][istr] = noisehist->GetRMS();
            n_sum += anal->noise_[iapv][istr];
            n_sum2 += (anal->noise_[iapv][istr] * anal->noise_[iapv][istr]);
            if ( anal->noise_[iapv][istr] > n_max ) { n_max = anal->noise_[iapv][istr]; }
            if ( anal->noise_[iapv][istr] < n_min ) { n_min = anal->noise_[iapv][istr]; }        
          
    		// Clean up time.
    		delete gausFit;
            //delete fit;
            delete FitHisto;
            delete noisehist;
        }
    } // strip loop
    
    // Calc mean and rms for peds
    if ( !anal->peds_[iapv].empty() ) { 
    	p_sum /= static_cast<float>( anal->peds_[iapv].size() );
        p_sum2 /= static_cast<float>( anal->peds_[iapv].size() );
        anal->pedsMean_[iapv] = p_sum;
        anal->pedsSpread_[iapv] = sqrt( fabs(p_sum2 - p_sum*p_sum) );
    }
    
    // Calc mean and rms for noise
    if ( !anal->noise_[iapv].empty() ) { 
    	n_sum /= static_cast<float>( anal->noise_[iapv].size() );
        n_sum2 /= static_cast<float>( anal->noise_[iapv].size() );
        anal->noiseMean_[iapv] = n_sum;
        anal->noiseSpread_[iapv] = sqrt( fabs(n_sum2 - n_sum*n_sum) );
    }

    // Calc mean and rms for raw noise
    if ( !anal->raw_[iapv].empty() ) { 
    	r_sum /= static_cast<float>( anal->raw_[iapv].size() );
        r_sum2 /= static_cast<float>( anal->raw_[iapv].size() );
        anal->rawMean_[iapv] = r_sum;
        anal->rawSpread_[iapv] = sqrt( fabs(r_sum2 - r_sum*r_sum) );
    }
    
    // Set max and min values for peds, noise and raw noise
    if ( p_max > -1.*sistrip::maximum_ ) { anal->pedsMax_[iapv] = p_max; }
    if ( p_min < 1.*sistrip::maximum_ )  { anal->pedsMin_[iapv] = p_min; }
    if ( n_max > -1.*sistrip::maximum_ ) { anal->noiseMax_[iapv] = n_max; }
    if ( n_min < 1.*sistrip::maximum_ )  { anal->noiseMin_[iapv] = n_min; }
    if ( r_max > -1.*sistrip::maximum_ ) { anal->rawMax_[iapv] = r_max; }
    if ( r_min < 1.*sistrip::maximum_ )  { anal->rawMin_[iapv] = r_min; }
    
    // Set dead and noisy strips
    for ( uint16_t istr = 0; istr < 128; istr++ ) {
    	// Set the significance of the noise of each strip also compared to apv avg.
        anal->noiseSignif_[iapv][istr] = (anal->noise_[iapv][istr]-anal->noiseMean_[iapv])/anal->noiseSpread_[iapv];
    	
        if ( anal->noiseMin_[iapv] > sistrip::maximum_ || anal->noiseMax_[iapv] > sistrip::maximum_ ) { 
        	continue; 
        }
        // Strip Masking for Dead Strips
        if((anal->noise_[iapv][istr]-anal->noiseMean_[iapv])/anal->noiseSpread_[iapv] < -10){
        	anal->dead_[iapv].push_back(istr);
            // Outputs "Dead NoiseSigif Crate Slot Ring Addr ccuCh lldCh iStrip NumBinsADC"
            /*TH1D * noisehist = noise_histo->ProjectionX("projx", iapv*128 + istr + 1, iapv*128 + istr + 1);
            SiStripFecKey fec_key(anal->fecKey());
           	std::cout<<"Dead "<<(anal->noise_[iapv][istr]-anal->noiseMean_[iapv])/anal->noiseSpread_[iapv]
            <<" "<<fec_key.fecCrate()
            <<" "<<fec_key.fecSlot()
            <<" "<<fec_key.fecRing()
            <<" "<<fec_key.ccuAddr()
            <<" "<<fec_key.ccuChan()
            <<" "<<fec_key.lldChan()
            <<" "<<iapv*128+istr<<" ";
            for(int i = 0; i < noisehist->GetNbinsX();i++){
            	std::cout << noisehist->GetBinContent(i+1) << " ";
            }
            std::cout << std::endl;       
        	delete noisehist;*/
        }
        //Strip Masking for Non Gassian Strips which are also noisy.
        else if(anal->ksProb_[iapv][istr] < 1 && (anal->noiseSignif_[iapv][istr] > 5 || anal->noise_[iapv][istr] > 8)){
        	anal->noisy_[iapv].push_back(istr);
            // Outputs "Dead KsProb Crate Slot Ring Addr ccuCh lldCh iStrip NumBinsADC"
            /*TH1D * noisehist = noise_histo->ProjectionX("projx", iapv*128 + istr + 1, iapv*128 + istr + 1);
        	SiStripFecKey fec_key(anal->fecKey());
            std::cout<<"Noisy "<<anal->ksProb_[iapv][istr]/10000.
            <<" "<<fec_key.fecCrate()
            <<" "<<fec_key.fecSlot()
            <<" "<<fec_key.fecRing()
            <<" "<<fec_key.ccuAddr()
            <<" "<<fec_key.ccuChan()
            <<" "<<fec_key.lldChan()
            <<" "<<iapv*128+istr<<" ";
            for(int i = 0; i < noisehist->GetNbinsX();i++){
            	std::cout << noisehist->GetBinContent(i+1) << " ";
            }
        	std::cout << std::endl;
            delete noisehist;*/
        }  
    	
        //if ( anal->noise_[iapv][istr] < ( anal->noiseMean_[iapv] - deadStripMax_ * anal->noiseSpread_[iapv] ) ) {
		//	anal->dead_[iapv].push_back(istr);
        //} 
       	//else if ( anal->ksProb_[iapv][istr] <= 10 ) { //Masking using KSProb*1000
        //	anal->noisy_[iapv].push_back(istr);
        //    std::cout << "KSMasked"<< anal->detId() << std::endl;
        //}
    	//else if ( anal->noise_[iapv][istr] > ( anal->noiseMean_[iapv] + noisyStripMin_ * anal->noiseSpread_[iapv] ) ) {
		//else if (istr%10 == 0){
        //	anal->noisy_[iapv].push_back(istr); //Masking using number of significances of noise.
        //    std::cout << "SigMasked Strip: " << iapv*128+istr << " Det:"<< anal->detId() << " DCU:" << anal->dcuId() 
        //    << " Fed:"<< anal->fedKey() << " FEC:" << anal->fecKey() << std::endl;
       // }        
    }    
  } // apv loop
  //std::cout << std::endl;
}
