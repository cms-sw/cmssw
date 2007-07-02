#include "DQM/SiStripCommissioningAnalysis/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommissioningAnalysis/interface/Utilities.h"
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
OptoScanAnalysis::OptoScanAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"OptoScanAnalysis"),
    gain_(sistrip::invalid_), 
    bias_(4,sistrip::invalid_), 
    measGain_(4,sistrip::invalid_), 
    zeroLight_(4,sistrip::invalid_), 
    linkNoise_(4,sistrip::invalid_),
    liftOff_(4,sistrip::invalid_), 
    threshold_(4,sistrip::invalid_), 
    tickHeight_(4,sistrip::invalid_),
    g0d0_(0,""), 
    g0d1_(0,""), 
    g1d0_(0,""), 
    g1d1_(0,""), 
    g2d0_(0,""), 
    g2d1_(0,""), 
    g3d0_(0,""), 
    g3d1_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
OptoScanAnalysis::OptoScanAnalysis() 
  : CommissioningAnalysis("OptoScanAnalysis"),
    gain_(sistrip::invalid_), 
    bias_(4,sistrip::invalid_), 
    measGain_(4,sistrip::invalid_), 
    zeroLight_(4,sistrip::invalid_), 
    linkNoise_(4,sistrip::invalid_),
    liftOff_(4,sistrip::invalid_), 
    threshold_(4,sistrip::invalid_), 
    tickHeight_(4,sistrip::invalid_),
    g0d0_(0,""), 
    g0d1_(0,""), 
    g1d0_(0,""), 
    g1d1_(0,""), 
    g2d0_(0,""), 
    g2d1_(0,""), 
    g3d0_(0,""), 
    g3d1_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::print( std::stringstream& ss, uint32_t gain ) { 
  if ( gain >= 4 ) { gain = gain_; }
  header( ss );
  ss << " Optimum LLD gain setting : " << gain_ << "\n"
     << " LLD bias setting         : " << bias_[gain] << "\n"
     << " Measured gain [V/V]      : " << measGain_[gain] << "\n"
     << " 'Zero light' level [adc] : " << zeroLight_[gain] << "\n"
     << " Link noise [adc]         : " << linkNoise_[gain] << "\n"
     << " Baseline 'lift off' [mA] : " << liftOff_[gain] << "\n"
     << " Laser threshold [mA]     : " << threshold_[gain] << "\n"
     << " Tick mark height [adc]   : " << tickHeight_[gain] << "\n";
}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::reset() {
  gain_       = sistrip::invalid_; 
  bias_       = VInts(4,sistrip::invalid_); 
  measGain_   = VFloats(4,sistrip::invalid_); 
  zeroLight_  = VFloats(4,sistrip::invalid_); 
  linkNoise_  = VFloats(4,sistrip::invalid_);
  liftOff_    = VFloats(4,sistrip::invalid_); 
  threshold_  = VFloats(4,sistrip::invalid_); 
  tickHeight_ = VFloats(4,sistrip::invalid_);
  g0d0_ = Histo(0,""); 
  g0d1_ = Histo(0,""); 
  g1d0_ = Histo(0,""); 
  g1d1_ = Histo(0,""); 
  g2d0_ = Histo(0,""); 
  g2d1_ = Histo(0,""); 
  g3d0_ = Histo(0,""); 
  g3d1_ = Histo(0,"");
}
  
// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check
  if ( histos.size() != 8 ) {
    edm::LogWarning(mlCommissioning_) 
      << "[" << myName() << "::" << __func__ << "]"
	 << " Unexpected number of histograms: " 
      << histos.size();
  }
  
  // Extract FED key from histo title
  if ( !histos.empty() ) extractFedKey( histos.front() );

  // Extract
  std::vector<TH1*>::const_iterator ihis = histos.begin();
  for ( ; ihis != histos.end(); ihis++ ) {
    
    // Check pointer
    if ( !(*ihis) ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to histogram!";
      continue;
    }
    
    // Check name
    SiStripHistoTitle title( (*ihis)->GetName() );
    if ( title.runType() != sistrip::OPTO_SCAN ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected commissioning task: "
	<< SiStripEnumsAndStrings::runType(title.runType());
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

    // Store opto scan histos
    if ( digital == 0 ) { 
      if ( gain == 0 ) { 
	g0d0_.first = *ihis; 
	g0d0_.second = (*ihis)->GetName();
      } else if ( gain == 1 ) { 
	g1d0_.first = *ihis; 
	g1d0_.second = (*ihis)->GetName();
      } else if ( gain == 2 ) { 
	g2d0_.first = *ihis; 
	g2d0_.second = (*ihis)->GetName();
      } else if ( gain == 3 ) { 
	g3d0_.first = *ihis; 
	g3d0_.second = (*ihis)->GetName();
      } else {
	edm::LogWarning(mlCommissioning_) 
	  << "[" << myName() << "::" << __func__ << "]"
	  << " Unexpected gain setting! (" << gain << ")";
      }
    } else if ( digital == 1 ) { 
      if ( gain == 0 ) { 
	g0d1_.first = *ihis; 
	g0d1_.second = (*ihis)->GetName();
      } else if ( gain == 1 ) { 
	g1d1_.first = *ihis; 
	g1d1_.second = (*ihis)->GetName();
      } else if ( gain == 2 ) { 
	g2d1_.first = *ihis; 
	g2d1_.second = (*ihis)->GetName();
      } else if ( gain == 3 ) { 
	g3d1_.first = *ihis; 
	g3d1_.second = (*ihis)->GetName();
      } else {
	edm::LogWarning(mlCommissioning_) 
	  << "[" << myName() << "::" << __func__ << "]"
	  << " Unexpected gain setting! (" << gain << ")";
      }
    } else {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unexpected ditigal setting! (" << digital << ")";
    }
    
  }

}

// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::analyse() { 
  
  if (0) { deprecated(); return; }
  
  // Iterate through four gain settings
  for ( uint16_t igain = 0; igain < 4; igain++ ) {
    
    // Select histos appropriate for gain setting
    TH1* base_his = 0;
    TH1* peak_his = 0;
    if      ( igain == 0 ) { base_his = g0d0_.first; peak_his = g0d1_.first; }
    else if ( igain == 1 ) { base_his = g1d0_.first; peak_his = g1d1_.first; }
    else if ( igain == 2 ) { base_his = g2d0_.first; peak_his = g2d1_.first; }
    else if ( igain == 3 ) { base_his = g3d0_.first; peak_his = g3d1_.first; }

    // Check for valid pointers to histograms
    if ( !peak_his ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to 'peak' histogram for gain setting: " 
	<< igain;
      continue;
    }
    if ( !base_his ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to 'base' histogram for gain setting: " 
	<< igain;
      continue;
    }

    // Extract TProfile histograms
    TProfile* base_histo = dynamic_cast<TProfile*>(base_his);
    TProfile* peak_histo = dynamic_cast<TProfile*>(peak_his);

    // Check for valid pointers to histograms
    if ( !peak_histo ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to 'peak' TProfile histogram for gain setting: " 
	<< igain;
      continue;
    }
    if ( !base_histo ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " NULL pointer to 'base' TProfile histogram for gain setting: " 
	<< igain;
      continue;
    }
    
    // Check histogram binning
    uint16_t nbins = static_cast<uint16_t>( peak_histo->GetNbinsX() );
    if ( static_cast<uint16_t>( base_histo->GetNbinsX() ) != nbins ) {
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Inconsistent number of bins for 'peak' and 'base' histograms: "
	<< nbins << " and " << static_cast<uint16_t>( base_histo->GetNbinsX() );
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
    float peak_max = -1.*sistrip::invalid_;
    float peak_min =  1.*sistrip::invalid_;
    float base_max = -1.*sistrip::invalid_;
    float base_min =  1.*sistrip::invalid_;

    // Transfer histogram contents/errors/stats to containers
    for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
      // Peak histogram
//       LogTrace(mlCommissioning_) << "ibin: " << ibin
// 	   << " peak: " << peak_histo->GetBinContent(ibin+1)
// 	   << " base: " << base_histo->GetBinContent(ibin+1)
// 	  ;
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
    }
    
    // Find "zero light" level and error
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
      edm::LogWarning(mlCommissioning_) 
	<< "[" << myName() << "::" << __func__ << "]"
	<< " Unable to find zero_light level."
	<< " No entries in histogram.";
      return;
    }
//     LogTrace(mlCommissioning_) << " zero_light_level: " << zero_light_level
// 	 << " zero_light_error: " << zero_light_error
// 	 << " zero_light_thres: " << zero_light_thres
// 	;

    // Find range of base histogram
    float base_range = base_max - base_min;

    // Find overlapping max/min region that constrains range of linear fit
    float max = peak_max < base_max ? peak_max : base_max;
    float min = peak_min > base_min ? peak_min : base_min;
    float range = max - min;

//     LogTrace(mlCommissioning_) << " peak_max: " << peak_max
// 	 << " peak_min: " << peak_min
// 	 << " base_max: " << base_max
// 	 << " base_min: " << base_min 
// 	;
//     LogTrace(mlCommissioning_) << " max: " << max
// 	 << " min: " << min
// 	 << " range: " << range
// 	 << " base_range: " << base_range
// 	;
      
    // Container identifying whether samples from 'base' histo are above "zero light" 
    std::vector<bool> above_zero_light;
    above_zero_light.resize(3,true);
    
    // Linear fits to top of peak and base curves and one to bottom of base curve
    sistrip::LinearFit peak_high;
    sistrip::LinearFit base_high;
    sistrip::LinearFit base_low;

//     LogTrace(mlCommissioning_) << " lower: " << min + 0.2*range
// 	 << " upper: " << min + 0.8*range
// 	 << " LOWlower: " << base_min + 0.2*base_range
// 	 << " LOWupper: " << base_min + 0.6*base_range
// 	;

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
	  //LogTrace(mlCommissioning_) << "peak: ";
	  peak_high.add( ibin, peak_contents[ibin], peak_entries[ibin] );
	}
      }
      // Linear fit to base histogram
      if ( base_entries[ibin] &&
	   base_contents[ibin] > ( min + 0.2*range ) &&
	   base_contents[ibin] < ( min + 0.8*range ) ) {
	if ( !base_bin ) { base_bin = ibin; }
	if ( ( ibin - base_bin ) < 10 ) { 
	  //LogTrace(mlCommissioning_) << "base: ";
	  base_high.add( ibin, base_contents[ibin], base_entries[ibin] );
	}
      }
      // Low linear fit to base histogram
      if ( base_entries[ibin] &&
	   //above_zero_light[0] && above_zero_light[1] && above_zero_light[2] && 
	   base_contents[ibin] > ( base_min + 0.2*base_range ) &&
	   base_contents[ibin] < ( base_min + 0.6*base_range ) ) { 
	if ( !low_bin ) { low_bin = ibin; }
	if ( ( ibin - low_bin ) < 10 ) { 
	  //LogTrace(mlCommissioning_) << "low: ";
	  base_low.add( ibin, base_contents[ibin], base_entries[ibin] );
	}
      }
      
    }

//     LogTrace(mlCommissioning_) << " peak_bin: " << peak_bin
// 	 << " base_bin: " << base_bin
// 	 << " low_bin: " << low_bin
// 	;

    // Extract width between two curves at midpoint within range
    float mid = min + 0.5*range;
    sistrip::LinearFit::Params peak_params;
    sistrip::LinearFit::Params base_params;
    peak_high.fit( peak_params );
    base_high.fit( base_params );
    float peak_pos = ( mid - peak_params.a_ ) / peak_params.b_;
    float base_pos = ( mid - base_params.a_ ) / base_params.b_;
    float width = base_pos - peak_pos;

//     LogTrace(mlCommissioning_) << " peak fit to " << peak_params.n_ << " points:"
// 	 << " peak intercept: " << peak_params.a_ << "+/-" << peak_params.erra_
// 	 << " peak gradient: " << peak_params.b_ << "+/-" << peak_params.errb_
// 	;
//     LogTrace(mlCommissioning_) << " base fit to " << base_params.n_ << " points:"
// 	 << " base intercept: " << base_params.a_ << "+/-" << base_params.erra_
// 	 << " base gradient: " << base_params.b_ << "+/-" << base_params.errb_
// 	;

//     LogTrace(mlCommissioning_) << " peak_pos: " << peak_pos
// 	 << " base_pos: " << base_pos
// 	 << " width: " << width
// 	;

    // Extrapolate to zero light to find "lift off"
    sistrip::LinearFit::Params low_params;
    base_low.fit( low_params );
    float lift_off = ( zero_light_level - low_params.a_ ) / low_params.b_;
    
//     LogTrace(mlCommissioning_) << " low fit to " << low_params.n_ << " points:"
// 	 << " low intercept: " << low_params.a_ << "+/-" << low_params.erra_
// 	 << " low gradient: " << low_params.b_ << "+/-" << low_params.errb_
// 	;

//     LogTrace(mlCommissioning_) << " lift off: " << lift_off;
    
    // ---------- Set all parameters ----------

    // Check "lift off" value and set bias setting accordingly
    if ( lift_off <= sistrip::maximum_ ) {
      bias_[igain] = static_cast<uint16_t>( lift_off ) + 2;
    } else { bias_[igain] = 0; } //@@ "default" should be what?

    // Set "zero light" level and link noise
    zeroLight_[igain] = zero_light_level;
    linkNoise_[igain] = zero_light_error;
    
    // Calculate "lift off" and laser threshold (in mA)
    liftOff_[igain] = 0.45 * lift_off;
    threshold_[igain] = 0.45 * ( lift_off - width/2. );
    
    // Calculate tick mark height
    if ( low_params.b_ <= sistrip::maximum_ &&
	 width <= sistrip::maximum_ ) {
      tickHeight_[igain] = width * low_params.b_;
    }

    // Set measured gain 
    if ( tickHeight_[igain] < sistrip::invalid_-1. ) {
      float adc_gain = 1.024 / 1024.; // Peak-to-peak voltage for FED ADC [V/adc] 
      measGain_[igain] = tickHeight_[igain] * adc_gain / 0.800;
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
      gain_ = static_cast<uint16_t>( igain );
      diff_in_gain = fabs( measGain_[igain] - target_gain );
    }
    
  } 

  // Check optimum gain setting
  if ( gain_ > sistrip::maximum_ ) { gain_ = 0; }

}


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// 
void OptoScanAnalysis::deprecated() { 
  
  std::vector<const TProfile*> histos; 
  std::vector<float> monitorables;

  for ( uint16_t igain = 0; igain < 4; igain++ ) {

    histos.clear();
    if ( igain == 0 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g0d0_.first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g0d1_.first) ) );
    } else if ( igain == 1 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g1d0_.first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g1d1_.first) ) );
    } else if ( igain == 2 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g2d0_.first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g2d1_.first) ) );
    } else if ( igain == 3 ) {
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g3d0_.first) ) );
      histos.push_back( const_cast<const TProfile*>( dynamic_cast<TProfile*>(g3d1_.first) ) );
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







