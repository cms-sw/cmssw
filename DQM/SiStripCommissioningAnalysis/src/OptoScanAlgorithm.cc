#include "DQM/SiStripCommissioningAnalysis/interface/OptoScanAlgorithm.h"
#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQM/SiStripCommissioningAnalysis/src/Utility.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
OptoScanAlgorithm::OptoScanAlgorithm( const edm::ParameterSet & pset, OptoScanAnalysis* const anal ) 
  : CommissioningAlgorithm(anal),
    histos_( 4, std::vector<Histo>( 3, Histo(0,"") ) ),
    targetGain_(pset.getParameter<double>("TargetGain"))
{
  edm::LogInfo(mlCommissioning_)
    << "[PedestalsAlgorithm::" << __func__ << "]"
    << " Set target gain to: " << targetGain_;
}
  
// ----------------------------------------------------------------------------
// 
void OptoScanAlgorithm::extract( const std::vector<TH1*>& histos ) { 

  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[OptoScanAlgorithm::" << __func__ << "]"
      << " NULL pointer to Analysis object!";
    return; 
  }

  // Check number of histograms
  if ( histos.size() != 12 ) {
    anal()->addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) anal()->fedKey( extractFedKey( histos.front() ) );

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }

    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::OPTO_SCAN ) {
      anal()->addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    // Extract gain setting and digital high/low info
    uint16_t gain = sistrip::invalid_; 
    if ( title.extraInfo().find(sistrip::extrainfo::gain_) != std::string::npos ) {
      std::stringstream ss;
      ss << title.extraInfo().substr( title.extraInfo().find(sistrip::extrainfo::gain_) + (sizeof(sistrip::extrainfo::gain_) - 1), 1 );
      ss >> std::dec >> gain;
    }
    uint16_t digital = sistrip::invalid_; 
    if ( title.extraInfo().find(sistrip::extrainfo::digital_) != std::string::npos ) {
      std::stringstream ss;
      ss << title.extraInfo().substr( title.extraInfo().find(sistrip::extrainfo::digital_) + (sizeof(sistrip::extrainfo::digital_) - 1), 1 );
      ss >> std::dec >> digital;
    }
    bool baseline_rms = false;
    if ( title.extraInfo().find(sistrip::extrainfo::baselineRms_) != std::string::npos ) {
      baseline_rms = true;
    }
    
    if ( gain <= 3 ) { 
      if ( digital <= 1 ) {
	histos_[gain][digital].first = *ihis; 
	histos_[gain][digital].second = (*ihis)->GetName();
      } else if ( baseline_rms ) {
	histos_[gain][2].first = *ihis; 
	histos_[gain][2].second = (*ihis)->GetName();
      } else {
	anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
      }
    } else {
      anal()->addErrorCode(sistrip::unexpectedExtraInfo_);
    }
    
  }

}

// ----------------------------------------------------------------------------
// 
void OptoScanAlgorithm::analyse() { 
  
  if ( !anal() ) {
    edm::LogWarning(mlCommissioning_)
      << "[OptoScanAlgorithm::" << __func__ << "]"
      << " NULL pointer to base Analysis object!";
    return; 
  }
  
  CommissioningAnalysis* tmp = const_cast<CommissioningAnalysis*>( anal() );
  OptoScanAnalysis* anal = dynamic_cast<OptoScanAnalysis*>( tmp );
  if ( !anal ) {
    edm::LogWarning(mlCommissioning_)
      << "[OptoScanAlgorithm::" << __func__ << "]"
      << " NULL pointer to derived Analysis object!";
    return; 
  }
  
  // Iterate through four gain settings
  for ( uint16_t igain = 0; igain < 4; igain++ ) {
    
    // Select histos appropriate for gain setting
    TH1* base_his = histos_[igain][0].first; 
    TH1* peak_his = histos_[igain][1].first;
    TH1* noise_his = histos_[igain][2].first;

    if ( !base_his ) {
      anal->addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    if ( !peak_his ) {
      anal->addErrorCode(sistrip::nullPtr_);
      return;
    }

    if ( !noise_his ) {
      anal->addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    TProfile* base_histo = dynamic_cast<TProfile*>(base_his);
    if ( !base_histo ) {
      anal->addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    TProfile* peak_histo = dynamic_cast<TProfile*>(peak_his);
    if ( !peak_histo ) {
      anal->addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    TProfile* noise_histo = dynamic_cast<TProfile*>(noise_his);
    if ( !noise_histo ) {
      anal->addErrorCode(sistrip::nullPtr_);
      return;
    }

    // Check histogram binning
    uint16_t nbins = static_cast<uint16_t>( peak_histo->GetNbinsX() );
    if ( static_cast<uint16_t>( base_histo->GetNbinsX() ) != nbins ) {
      anal->addErrorCode(sistrip::numberOfBins_);
      if ( static_cast<uint16_t>( base_histo->GetNbinsX() ) < nbins ) {
	nbins = static_cast<uint16_t>( base_histo->GetNbinsX() );
      }
    }

    // Some containers
    std::vector<float> peak_contents(0);
    std::vector<float> peak_errors(0);
    std::vector<float> peak_entries(0);
    std::vector<float> base_contents(0);
    std::vector<float> base_errors(0);
    std::vector<float> base_entries(0);
    std::vector<float> noise_contents(0);
    std::vector<float> noise_errors(0);
    std::vector<float> noise_entries(0);
    float peak_max = -1.*sistrip::invalid_;
    float peak_min =  1.*sistrip::invalid_;
    float base_max = -1.*sistrip::invalid_;
    float base_min =  1.*sistrip::invalid_;
    float noise_max = -1.*sistrip::invalid_;
    float noise_min =  1.*sistrip::invalid_;

    // Transfer histogram contents/errors/stats to containers
    for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {

      // Peak histogram
      peak_contents.push_back( peak_histo->GetBinContent(ibin+1) );
      peak_errors.push_back( peak_histo->GetBinError(ibin+1) );
      peak_entries.push_back( peak_histo->GetBinEntries(ibin+1) );
      if ( peak_entries[ibin] ) { 
	if ( peak_contents[ibin] > peak_max ) { peak_max = peak_contents[ibin]; }
	if ( peak_contents[ibin] < peak_min && ibin ) { peak_min = peak_contents[ibin]; }
      }

      // Base histogram
      base_contents.push_back( base_histo->GetBinContent(ibin+1) );
      base_errors.push_back( base_histo->GetBinError(ibin+1) );
      base_entries.push_back( base_histo->GetBinEntries(ibin+1) );
      if ( base_entries[ibin] ) { 
	if ( base_contents[ibin] > base_max ) { base_max = base_contents[ibin]; }
	if ( base_contents[ibin] < base_min && ibin ) { base_min = base_contents[ibin]; }
      }

      // Noise histogram
      noise_contents.push_back( noise_histo->GetBinContent(ibin+1) );
      noise_errors.push_back( noise_histo->GetBinError(ibin+1) );
      noise_entries.push_back( noise_histo->GetBinEntries(ibin+1) );
      if ( noise_entries[ibin] ) { 
	if ( noise_contents[ibin] > noise_max ) { noise_max = noise_contents[ibin]; }
	if ( noise_contents[ibin] < noise_min && ibin ) { noise_min = noise_contents[ibin]; }
      }

    }
    
    // Find "zero light" level and error
    //@@ record bias setting used for zero light level
    //@@ zero light error changes wrt gain setting ???
    float zero_light_level = sistrip::invalid_;
    float zero_light_error = sistrip::invalid_;
    for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
      if ( base_entries[ibin] ) { 
	zero_light_level = base_contents[ibin];
	zero_light_error = base_errors[ibin];
	break;
      }
    }

    float zero_light_thres = sistrip::invalid_;
    if ( zero_light_level <= sistrip::maximum_ && 
	 zero_light_error <= sistrip::maximum_ ) { 
      zero_light_thres = zero_light_level + 5. * zero_light_error;
    } else {
      std::stringstream ss;
      ss << sistrip::invalidZeroLightLevel_ << "ForGain" << igain;
      anal->addErrorCode( ss.str() );
      continue;
    }
    
    // Find range of base histogram
    float base_range = base_max - base_min;

    // Find overlapping max/min region that constrains range of linear fit
    float max = peak_max < base_max ? peak_max : base_max;
    float min = peak_min > base_min ? peak_min : base_min;
    float range = max - min;

    // Container identifying whether samples from 'base' histo are above "zero light" 
    std::vector<bool> above_zero_light;
    above_zero_light.resize(3,true);
    
    // Linear fits to top of peak and base curves and one to bottom of base curve
    sistrip::LinearFit peak_high;
    sistrip::LinearFit base_high;
    sistrip::LinearFit base_low;

    // Iterate through histogram bins
    uint16_t peak_bin = 0;
    uint16_t base_bin = 0;
    uint16_t low_bin = 0;
    for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
      
      // Record whether consecutive samples from 'base' histo are above the "zero light" level
      if ( base_entries[ibin] ) {
	above_zero_light.erase( above_zero_light.begin() );
	if ( base_contents[ibin] > zero_light_thres ) { above_zero_light.push_back( true ); }
	else { above_zero_light.push_back( false ); }
	if ( above_zero_light.size() != 3 ) { above_zero_light.resize(3,false); } 
      }
      
      // Linear fit to peak histogram
      if ( peak_entries[ibin] &&
	   peak_contents[ibin] > ( min + 0.2*range ) &&
	   peak_contents[ibin] < ( min + 0.8*range ) ) {
	if ( !peak_bin ) { peak_bin = ibin; }
	if ( ( ibin - peak_bin ) < 10 ) { 
	  peak_high.add( ibin, peak_contents[ibin] ); //@@ should weight using bin error or bin contents (sqrt(N)/N)
	}
      }
      // Linear fit to base histogram
      if ( base_entries[ibin] &&
	   base_contents[ibin] > ( min + 0.2*range ) &&
	   base_contents[ibin] < ( min + 0.8*range ) ) {
	if ( !base_bin ) { base_bin = ibin; }
	if ( ( ibin - base_bin ) < 10 ) { 
	  base_high.add( ibin, base_contents[ibin] ); //@@ should weight using bin error or bin contents (sqrt(N)/N)
	}
      }
      // Low linear fit to base histogram
      if ( base_entries[ibin] &&
	   //@@ above_zero_light[0] && above_zero_light[1] && above_zero_light[2] && 
	   base_contents[ibin] > ( base_min + 0.2*base_range ) &&
	   base_contents[ibin] < ( base_min + 0.6*base_range ) ) { 
	if ( !low_bin ) { low_bin = ibin; }
	if ( ( ibin - low_bin ) < 10 ) { 
	  base_low.add( ibin, base_contents[ibin] ); //@@ should weight using bin error or bin contents (sqrt(N)/N)
	}
      }
      
    }
    
    // Extract width between two curves at midpoint within range
    float mid = min + 0.5*range;
    sistrip::LinearFit::Params peak_params;
    sistrip::LinearFit::Params base_params;
    peak_high.fit( peak_params );
    base_high.fit( base_params );

    float peak_pos = sistrip::invalid_;
    float base_pos = sistrip::invalid_;
    float width    = sistrip::invalid_;
    if ( peak_params.b_ > 0. ) {
      peak_pos = ( mid - peak_params.a_ ) / peak_params.b_;
    }
    if ( base_params.b_ > 0. ) {
      base_pos = ( mid - base_params.a_ ) / base_params.b_;
    }
    if ( base_pos < sistrip::valid_ &&
	 peak_pos < sistrip::valid_ ) {
      width = base_pos - peak_pos;
    }

    // Extrapolate to zero light to find "lift off"
    sistrip::LinearFit::Params low_params;
    base_low.fit( low_params );
    float lift_off = sistrip::invalid_;
    if ( low_params.b_ > 0. ) {
      lift_off = ( zero_light_level - low_params.a_ ) / low_params.b_;
    }
    
    // ---------- Set all parameters ----------
    
    // Slope of baseline
    if ( low_params.b_ > 0. ) {
      anal->baseSlope_[igain] = low_params.b_;
    } 
    
    // Check "lift off" value and set bias setting accordingly
    if ( lift_off <= sistrip::maximum_ ) {
      anal->bias_[igain] = static_cast<uint16_t>( lift_off ) + 2;
    } else { anal->bias_[igain] = OptoScanAnalysis::defaultBiasSetting_; } 
    
    // Calculate "lift off" (in mA)
    if ( lift_off <= sistrip::maximum_ ) {
      anal->liftOff_[igain] = 0.45 * lift_off;
    }
    
    // Calculate laser threshold (in mA)
    if ( width < sistrip::invalid_ ) {
      anal->threshold_[igain] = 0.45 * ( lift_off - width/2. );
    }

    // Set "zero light" level
    anal->zeroLight_[igain] = zero_light_level;
    
    // Set link noise
    uint16_t bin_number = sistrip::invalid_;
    if ( anal->threshold_[igain] < sistrip::valid_ ) {
      // Old calculation, used in commissioning in 2008
      //   always leads to zero link noise
      //   bin_number = static_cast<uint16_t>( anal->threshold_[igain] / 0.45 ); 
      // New calculation asked by Karl et al, for commissioning in 2009
      bin_number = (uint16_t) (lift_off + width / 3.);
    }
    if ( bin_number < sistrip::valid_ ) {
      if ( bin_number < noise_contents.size() ) { 
	anal->linkNoise_[igain] = noise_contents[bin_number]; 
      } else { anal->addErrorCode(sistrip::unexpectedBinNumber_); }
    }
      
    // Calculate tick mark height
    if ( low_params.b_ <= sistrip::maximum_ &&
	 width <= sistrip::maximum_ ) {
      anal->tickHeight_[igain] = width * low_params.b_;
    }
      
    // Set measured gain 
    if ( anal->tickHeight_[igain] < sistrip::invalid_-1. ) {
      anal->measGain_[igain] = anal->tickHeight_[igain] * OptoScanAnalysis::fedAdcGain_ / 0.800;
    } else { anal->measGain_[igain] = sistrip::invalid_; }
      
  } // gain loop

  // Iterate through four gain settings and identify optimum gain setting
  const float target_gain = targetGain_; //0.863; // changed from 0.8 to avoid choice of low tickheights (Xtof, SL, 15/6/2009)

  float diff_in_gain = sistrip::invalid_;
  for ( uint16_t igain = 0; igain < 4; igain++ ) {
    
    // Check for sensible gain value
    if ( anal->measGain_[igain] > sistrip::maximum_ ) { continue; }
    
    // Find optimum gain setting
    if ( fabs( anal->measGain_[igain] - target_gain ) < diff_in_gain ) {
      anal->gain_ = igain;
      diff_in_gain = fabs( anal->measGain_[igain] - target_gain );
    }
    
  } 
  
  // Check optimum gain setting
  if ( anal->gain_ > sistrip::maximum_ ) { anal->gain_ = OptoScanAnalysis::defaultGainSetting_; }
  
}

// ----------------------------------------------------------------------------
// 
CommissioningAlgorithm::Histo OptoScanAlgorithm::histo( const uint16_t& gain, 
							const uint16_t& digital_level ) const {
  if ( gain <= 3 && digital_level <= 1 ) { return histos_[gain][digital_level]; }
  else { return Histo(0,""); }
}
