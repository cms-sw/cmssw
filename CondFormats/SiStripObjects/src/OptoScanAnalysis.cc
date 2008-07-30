#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
const uint16_t OptoScanAnalysis::defaultGainSetting_ = sistrip::invalid_; //@@ 

// ----------------------------------------------------------------------------
// 
const uint16_t OptoScanAnalysis::defaultBiasSetting_ = sistrip::invalid_; //@@ 

// ----------------------------------------------------------------------------
// 
const float OptoScanAnalysis::fedAdcGain_ = 1.024 / 1024.; // [V/ADC]

// ----------------------------------------------------------------------------
// 
OptoScanAnalysis::OptoScanAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,sistrip::optoScanAnalysis_),
    gain_(sistrip::invalid_), 
    bias_(4,sistrip::invalid_), 
    measGain_(4,sistrip::invalid_), 
    zeroLight_(4,sistrip::invalid_), 
    linkNoise_(4,sistrip::invalid_),
    liftOff_(4,sistrip::invalid_), 
    threshold_(4,sistrip::invalid_), 
    tickHeight_(4,sistrip::invalid_),
    baseSlope_(4,sistrip::invalid_),
    histos_( 4, std::vector<Histo>( 3, Histo(0,"") ) )
{;}

// ----------------------------------------------------------------------------
// 
OptoScanAnalysis::OptoScanAnalysis() 
  : CommissioningAnalysis(sistrip::optoScanAnalysis_),
    gain_(sistrip::invalid_), 
    bias_(4,sistrip::invalid_), 
    measGain_(4,sistrip::invalid_), 
    zeroLight_(4,sistrip::invalid_), 
    linkNoise_(4,sistrip::invalid_),
    liftOff_(4,sistrip::invalid_), 
    threshold_(4,sistrip::invalid_), 
    tickHeight_(4,sistrip::invalid_),
    baseSlope_(4,sistrip::invalid_),
    histos_( 4, std::vector<Histo>( 3, Histo(0,"") ) )
{;}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::reset() {
  gain_       = sistrip::invalid_; 
  bias_       = VInt(4,sistrip::invalid_); 
  measGain_   = VFloat(4,sistrip::invalid_); 
  zeroLight_  = VFloat(4,sistrip::invalid_); 
  linkNoise_  = VFloat(4,sistrip::invalid_);
  liftOff_    = VFloat(4,sistrip::invalid_); 
  threshold_  = VFloat(4,sistrip::invalid_); 
  tickHeight_ = VFloat(4,sistrip::invalid_);
  baseSlope_  = VFloat(4,sistrip::invalid_);
  histos_.clear();
  histos_.resize( 4, std::vector<Histo>( 3, Histo(0,"") ) );
}
  
// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check number of histograms
  if ( histos.size() != 12 ) {
    addErrorCode(sistrip::numberOfHistos_);
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) extractFedKey( histos.front() );

  // Extract histograms
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check for NULL pointer
    if ( !(*ihis) ) { continue; }

    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::OPTO_SCAN ) {
      addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    // Extract gain setting and digital high/low info
    uint16_t gain = sistrip::invalid_; 
    if ( title.extraInfo().find(sistrip::gain_) != std::string::npos ) {
      std::stringstream ss;
      ss << title.extraInfo().substr( title.extraInfo().find(sistrip::gain_) + sistrip::gain_.size(), 1 );
      ss >> std::dec >> gain;
    }
    uint16_t digital = sistrip::invalid_; 
    if ( title.extraInfo().find(sistrip::digital_) != std::string::npos ) {
      std::stringstream ss;
      ss << title.extraInfo().substr( title.extraInfo().find(sistrip::digital_) + sistrip::digital_.size(), 1 );
      ss >> std::dec >> digital;
    }
    bool baseline_rms = false;
    if ( title.extraInfo().find(sistrip::baselineRms_) != std::string::npos ) {
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
	addErrorCode(sistrip::unexpectedExtraInfo_);
      }
    } else {
      addErrorCode(sistrip::unexpectedExtraInfo_);
    }
    
  }

}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::analyse() { 

  // Use deprecated method 
  if (0) { 
    deprecated(); 
    return; 
  }
  
  // Iterate through four gain settings
  for ( uint16_t igain = 0; igain < 4; igain++ ) {
    
    // Select histos appropriate for gain setting
    TH1* base_his = histos_[igain][0].first; 
    TH1* peak_his = histos_[igain][1].first;
    TH1* noise_his = histos_[igain][2].first;

    if ( !base_his ) {
      addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    if ( !peak_his ) {
      addErrorCode(sistrip::nullPtr_);
      return;
    }

    if ( !noise_his ) {
      addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    TProfile* base_histo = dynamic_cast<TProfile*>(base_his);
    if ( !base_histo ) {
      addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    TProfile* peak_histo = dynamic_cast<TProfile*>(peak_his);
    if ( !peak_histo ) {
      addErrorCode(sistrip::nullPtr_);
      return;
    }
    
    TProfile* noise_histo = dynamic_cast<TProfile*>(noise_his);
    if ( !noise_histo ) {
      addErrorCode(sistrip::nullPtr_);
      return;
    }

    // Check histogram binning
    uint16_t nbins = static_cast<uint16_t>( peak_histo->GetNbinsX() );
    if ( static_cast<uint16_t>( base_histo->GetNbinsX() ) != nbins ) {
      addErrorCode(sistrip::numberOfBins_);
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
      addErrorCode( ss.str() );
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
    float peak_pos = ( mid - peak_params.a_ ) / peak_params.b_;
    float base_pos = ( mid - base_params.a_ ) / base_params.b_;
    float width = base_pos - peak_pos;

    // Extrapolate to zero light to find "lift off"
    sistrip::LinearFit::Params low_params;
    base_low.fit( low_params );
    float lift_off = ( zero_light_level - low_params.a_ ) / low_params.b_;
    
    // ---------- Set all parameters ----------

    // Slope of baseline
    baseSlope_[igain] = low_params.b_;

    // Check "lift off" value and set bias setting accordingly
    if ( lift_off <= sistrip::maximum_ ) {
      bias_[igain] = static_cast<uint16_t>( lift_off ) + 2;
    } else { bias_[igain] = defaultBiasSetting_; } 
    
    // Calculate "lift off" and laser threshold (in mA)
    liftOff_[igain] = 0.45 * lift_off;
    threshold_[igain] = 0.45 * ( lift_off - width/2. );

    // Set "zero light" level and link noise
    zeroLight_[igain] = zero_light_level;
    uint16_t bin_number = static_cast<uint16_t>( threshold_[igain] / 0.45 ); 
    if ( bin_number < noise_contents.size() ) { linkNoise_[igain] = noise_contents[bin_number]; }
    else { addErrorCode(sistrip::unexpectedBinNumber_); }
    
    // Calculate tick mark height
    if ( low_params.b_ <= sistrip::maximum_ &&
	 width <= sistrip::maximum_ ) {
      tickHeight_[igain] = width * low_params.b_;
    }

    // Set measured gain 
    if ( tickHeight_[igain] < sistrip::invalid_-1. ) {
      measGain_[igain] = tickHeight_[igain] * fedAdcGain_ / 0.800;
    } else { measGain_[igain] = 0.; }
    
  } // gain loop

  // Iterate through four gain settings and identify optimum gain setting
  const float target_gain = 0.8;
  float diff_in_gain = sistrip::invalid_;
  for ( uint16_t igain = 0; igain < 4; igain++ ) {
    
    // Check for sensible gain value
    if ( measGain_[igain] > sistrip::maximum_ ) { continue; }
    
    // Find optimum gain setting
    if ( fabs( measGain_[igain] - target_gain ) < diff_in_gain ) {
      gain_ = igain;
      diff_in_gain = fabs( measGain_[igain] - target_gain );
    }
    
  } 

  // Check optimum gain setting
  if ( gain_ > sistrip::maximum_ ) { gain_ = defaultGainSetting_; }
  
}

// ----------------------------------------------------------------------------
// 
CommissioningAnalysis::Histo OptoScanAnalysis::histo( const uint16_t& gain, 
						      const uint16_t& digital_level ) const {
  if ( gain <= 3 && digital_level <= 1 ) { return histos_[gain][digital_level]; }
  else { return Histo(0,""); }
}

// ----------------------------------------------------------------------------
// 
bool OptoScanAnalysis::isValid() const {
  return ( gain_ < sistrip::maximum_ &&
	   bias_[gain_] < sistrip::maximum_ &&
	   getErrorCodes().empty() );
}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::summary( std::stringstream& ss ) const { 

  SiStripFecKey fec_key( fecKey() );
  SiStripFedKey fed_key( fedKey() );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );

  std::stringstream extra1,extra2,extra3; 
  extra1 << sistrip::gain_ << gain() << sistrip::digital_ << "0";
  extra2 << sistrip::gain_ << gain() << sistrip::digital_ << "1";
  extra3 << sistrip::gain_ << gain() << sistrip::baselineRms_;
  
  std::string title1 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::LLD_CHAN, 
					  fec_key.lldChan(),
					  extra1.str() ).title();
  std::string title2 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::LLD_CHAN, 
					  fec_key.lldChan(),
					  extra2.str() ).title();
  std::string title3 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::LLD_CHAN, 
					  fec_key.lldChan(),
					  extra3.str() ).title();
  
  ss << " Summary"
     << ":"
     << ( isValid() ? "Valid" : "Invalid" )
     << ":"
     << sistrip::controlView_ << ":"
     << fec_key.fecCrate() << "/" 
     << fec_key.fecSlot() << "/" 
     << fec_key.fecRing() << "/" 
     << fec_key.ccuAddr() << "/" 
     << fec_key.ccuChan() 
     << ":"
     << sistrip::dqmRoot_ << sistrip::dir_ 
     << "Collate" << sistrip::dir_ 
     << SiStripFecKey( fec_key.fecCrate(),
		       fec_key.fecSlot(), 
		       fec_key.fecRing(), 
		       fec_key.ccuAddr(), 
		       fec_key.ccuChan() ).path()
     << ":"
     << title1 << ";" << title2 << ";" << title3
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::print( std::stringstream& ss, uint32_t gain ) { 

  if ( gain >= 4 ) { gain = gain_; }

  if ( gain >= bias_.size() ) {
    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected gain setting: " << gain_;
    return;
  }

  header( ss );
  if ( gain_ > sistrip::maximum_ ) { 
    ss << " Warning: invalid gain setting!" << std::endl;
    ss << " (Monitorables below for gain setting " << gain << ")" << std::endl;
  }
  ss <<  std::fixed << std::setprecision(2)
     << " Optimum LLD gain setting : " << gain_ << std::endl
     << " LLD bias setting         : " << bias_[gain] << std::endl
     << " Measured gain      [V/V] : " << measGain_[gain] << std::endl
     << " Zero light level   [ADC] : " << zeroLight_[gain] << std::endl
     << " Link noise [ADC]         : " << linkNoise_[gain] << std::endl
     << " Baseline 'lift off' [mA] : " << liftOff_[gain] << std::endl
     << " Laser threshold     [mA] : " << threshold_[gain] << std::endl
     << " Tick mark height   [ADC] : " << tickHeight_[gain] << std::endl
     << " Baseline slope [ADC/I2C] : " << baseSlope_[gain] << std::endl
     << std::boolalpha 
     << " isValid                  : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << ")   : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;

}


// ---------- DEPRECATED METHODS BELOW ----------


// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::deprecated() { 
  
  std::vector<const TProfile*> histos; 
  std::vector<float> monitorables;

  for ( uint16_t igain = 0; igain < 4; igain++ ) {

    histos.clear();
    if ( igain == 0 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[0][0].first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[0][1].first) ) );
    } else if ( igain == 1 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[1][0].first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[1][1].first) ) );
    } else if ( igain == 2 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[2][0].first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[2][1].first) ) );
    } else if ( igain == 3 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[3][0].first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(histos_[3][1].first) ) );
    } 
    
    if ( !histos[0] ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to base histo!";
      continue;
    }
    if ( !histos[1] ) {
      edm::LogWarning(mlCommissioning_)
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to tick histo!";
      continue;
    }

    monitorables.clear();
    anal( histos, monitorables );
    
    bias_[igain] = static_cast<uint16_t>( monitorables[0] );
    measGain_[igain] = monitorables[1];
    
  }
 
  // Target gain
  float target_gain = 0.8;
  float diff_in_gain = sistrip::invalid_;
  
  // Iterate through four gain settings
  for ( uint16_t igain = 0; igain < 4; igain++ ) {

    // Check for sensible gain value
    if ( measGain_[igain] > sistrip::maximum_ ) { continue; }
    
    // Check for sensible bias value
    if ( bias_[igain] > sistrip::maximum_ ) { bias_[igain] = 30; } //@@ should be what???

    // Find optimum gain setting
    if ( fabs( measGain_[igain] - target_gain ) < diff_in_gain ) {
      gain_ = static_cast<uint16_t>( igain );
      diff_in_gain = fabs( measGain_[igain] - target_gain );
    }
    
  } 

  // Check optimum gain setting
  if ( gain_ > sistrip::maximum_ ) { gain_ = 0; }

}

// ----------------------------------------------------------------------------
//
void OptoScanAnalysis::anal( const std::vector<const TProfile*>& histos, 
			     std::vector<float>& monitorables ) {
  //LogDebug("Commissioning|Analysis") << "[OptoScanAnalysis::analysis]";
  
  //extract root histograms
  //check 
  if (histos.size() != 2) { 
    //     edm::LogError("Commissioning|Analysis") 
    //       << "[OptoScanAnalysis::analysis]: Requires \"const std::vector<const TH1F*>& \" argument to have size 2. Actual size: " 
    //       << histos.size() << ". Monitorables set to 0."; 
    monitorables.push_back(0); monitorables.push_back(0);
  }

  //relabel
  const TProfile* base = histos[0];
  const TProfile* peak = histos[1];

  if ( !base ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to base histo!";
  }
  if ( !peak ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
      << " NULL pointer to peak histo!";
  }
  if ( !base || !peak ) { return; }

  //define utility objects
  std::vector<float> second_deriv_base; second_deriv_base.reserve(44); second_deriv_base.resize(44,0.);
  std::pair< int, int > slope_edges_base; slope_edges_base.first = 5; slope_edges_base.second = 5;
  
  //calculate the 2nd derivative of the histos and find slope edges

  for (int k=5;k<45;k++) {

    //checks

    // if (!base->GetBinContent(k)) {LogTrace(mlCommissioning_) << "[OptoScanAnalysis::analysis]: Warning: Tick base has recorded value of 0 at bias: " << k - 1;}

    // if (!peak->GetBinContent(k)) { LogTrace(mlCommissioning_) << "[OptoScanAnalysis::analysis]: Warning: Tick peak has recorded value of 0 at bias: " << k - 1;}

    second_deriv_base[k - 1] = 
      base->GetBinContent(k + 1) - 
      2*(base->GetBinContent(k)) + 
      base->GetBinContent(k - 1);

    //find bins containing 2 peaks in 2nd deriv i.e."slope edges"
    
    if (second_deriv_base[slope_edges_base.first] < second_deriv_base[k - 1]) { slope_edges_base.first = k - 1;}
    if (second_deriv_base[slope_edges_base.second] > second_deriv_base[k - 1]) { slope_edges_base.second = k - 1; }
  }

  //check

  if (slope_edges_base.first > slope_edges_base.second) {
    //        LogDebug("Commissioning|Analysis") 
    // 	 << "[OptoScanAnalysis::analysis]: Warning: Maximum second derivative of tick base occurs at higher bias: " 
    // 	 << slope_edges_base.first <<  " than the minimum: " << slope_edges_base.second << ".";
  }

  //CALCULATE BIAS
  //find position of - first point after 2nd deriv max below 0.2 x max (for base) - and - first point before 2nd deriv min above 0.2 x min (for base and peak).

  while (fabs(second_deriv_base[slope_edges_base.second]) > fabs(0.2 * slope_edges_base.second)) {slope_edges_base.second--;}
  while (fabs(second_deriv_base[slope_edges_base.first]) > fabs(0.2 * slope_edges_base.first)) {slope_edges_base.first++;}
  
  float bias = (float)slope_edges_base.first;

  //CALCULATE GAIN
  //Find bias where the peak/base is closest to 300 (tunable)...

  float slope_grad_base = (float) (base->GetBinContent(slope_edges_base.second) - base->GetBinContent(slope_edges_base.first)) / (float)(slope_edges_base.second - slope_edges_base.first);

  unsigned short slope_centerx_base = 0, slope_centerx_peak = 0;

  for (unsigned short baseBias = 0; baseBias < 45; baseBias++) {
    if (fabs(base->GetBinContent((Int_t)(baseBias + 1)) - 300) < fabs(base->GetBinContent((Int_t)(slope_centerx_base)) - 300)) slope_centerx_base = baseBias;}

  for (unsigned short peakBias = 0; peakBias < 45; peakBias++) {
    if (fabs(peak->GetBinContent((Int_t)(peakBias + 1)) - 300) < fabs(peak->GetBinContent((Int_t)(slope_centerx_peak)) - 300)) slope_centerx_peak = peakBias;}
 
  //check
  if (((peak->GetBinContent((Int_t)(slope_centerx_peak)) - base->GetBinContent((Int_t)(slope_centerx_base)))/ (float)base->GetBinContent((Int_t)(slope_centerx_base))) > 0.1) { 
    //        LogDebug("Commissioning|Analysis") 
    // 	 << "[OptoScanAnalysis::analysis]: Warning: No tick height found to match tick base at 70% off its maximum (> 10% difference between histograms)."; 
  }

  //Gain
  float gain = (slope_centerx_base - slope_centerx_peak) * slope_grad_base / 800.;

  //set monitorables
  monitorables.clear();
  monitorables.push_back(bias);
  monitorables.push_back(gain);

}


// -------------------------------------
// ---------- Utility classes ----------
// -------------------------------------

// ----------------------------------------------------------------------------
// 
sistrip::LinearFit::LinearFit() 
  : x_(),
    y_(),
    e_(),
    ss_(0.),
    sx_(0.),
    sy_(0.) 
{ 
}

// ----------------------------------------------------------------------------
// 
void sistrip::LinearFit::add( const float& x,
			      const float& y ) {
  float e = 1.; // default
  x_.push_back(x);
  y_.push_back(y);
  e_.push_back(e);
  float wt = 1. / sqrt(e); 
  ss_ += wt;
  sx_ += x*wt;
  sy_ += y*wt;
}

// ----------------------------------------------------------------------------
// 
void sistrip::LinearFit::add( const float& x,
			      const float& y,
			      const float& e ) {
  if ( e > 0. ) { 
    x_.push_back(x);
    y_.push_back(y);
    e_.push_back(e);
    float wt = 1. / sqrt(e); 
    ss_ += wt;
    sx_ += x*wt;
    sy_ += y*wt;
  } 
}

// ----------------------------------------------------------------------------
// 
void sistrip::LinearFit::fit( Params& params ) {

  float s2 = 0.;
  float b = 0;
  for ( uint16_t i = 0; i < x_.size(); i++ ) {
    float t = ( x_[i] - sx_/ss_ ) / e_[i]; 
    s2 += t*t;
    b += t * y_[i] / e_[i];
  }
  
  // Set parameters
  params.n_ = x_.size();
  params.b_ = b / s2;
  params.a_ = ( sy_ - sx_ * params.b_ ) / ss_;
  params.erra_ = sqrt( ( 1. + (sx_*sx_) / (ss_*s2) ) / ss_ );
  params.errb_ = sqrt( 1. / s2 );
  
  /*
    params.chi2_ = 0.;
    *q=1.0;
    if (mwt == 0) {
    for (i=1;i<=ndata;i++)
    *chi2 += SQR(y[i]-(*a)-(*b)*x[i]);
    sigdat=sqrt((*chi2)/(ndata-2));
    *sigb *= sigdat;
    */    
  
}

// ----------------------------------------------------------------------------
// 
sistrip::MeanAndStdDev::MeanAndStdDev() 
  : s_(0.),
    x_(0.),
    xx_(0.),
    vec_()
{;}

// ----------------------------------------------------------------------------
// 
void sistrip::MeanAndStdDev::add( const float& x,
			 const float& e ) {
  if ( e > 0. ) { 
    float wt = 1. / sqrt(e); 
    s_ += wt;
    x_ += x*wt;
    xx_ += x*x*wt;
  } else {
    s_++;
    x_ += x;
    xx_ += x*x;
  }
  vec_.push_back(x);
}

// ----------------------------------------------------------------------------
// 
void sistrip::MeanAndStdDev::fit( Params& params ) {
  if ( s_ > 0. ) { 
    float m = x_/s_;
    float t = xx_/s_ - m*m;
    if ( t > 0. ) { t = sqrt(t); } 
    else { t = 0.; }
    params.mean_ = m;
    params.rms_  = t;
  }
  if ( !vec_.empty() ) {
    sort( vec_.begin(), vec_.end() );
    uint16_t index = vec_.size()%2 ? vec_.size()/2 : vec_.size()/2-1;
    params.median_ = vec_[index];
  }      
}






