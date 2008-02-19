#include "CondFormats/SiStripObjects/interface/FastFedCablingAnalysis.h"
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
const float FastFedCablingAnalysis::threshold_ = 100.; // [ADC]

// ----------------------------------------------------------------------------
// 
const float FastFedCablingAnalysis::dirtyThreshold_ = 800; // [ADC]

// ----------------------------------------------------------------------------
// 
const float FastFedCablingAnalysis::trimDacThreshold_ = 10; // [ADC]

// ----------------------------------------------------------------------------
// 
const uint16_t FastFedCablingAnalysis::nBitsForDcuId_ = 32;

// ----------------------------------------------------------------------------
// 
const uint16_t FastFedCablingAnalysis::nBitsForLldCh_ = 2;

// ----------------------------------------------------------------------------
// 
FastFedCablingAnalysis::FastFedCablingAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,sistrip::fastCablingAnalysis_),
    dcuHardId_(sistrip::invalid32_), 
    lldCh_(sistrip::invalid_),
    highMedian_(1.*sistrip::invalid_),
    highMean_(1.*sistrip::invalid_),
    highRms_(1.*sistrip::invalid_),
    lowMedian_(1.*sistrip::invalid_),
    lowMean_(1.*sistrip::invalid_),
    lowRms_(1.*sistrip::invalid_),
    range_(1.*sistrip::invalid_),
    midRange_(1.*sistrip::invalid_),
    max_(1.*sistrip::invalid_),
    min_(1.*sistrip::invalid_),
    histo_(0,"")
{
  fecKey( SiStripFecKey( sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_, 
			 sistrip::invalid_ ).key() );
  fedKey( key );
}

// ----------------------------------------------------------------------------
// 
FastFedCablingAnalysis::FastFedCablingAnalysis() 
  : CommissioningAnalysis(sistrip::fastCablingAnalysis_),
    dcuHardId_(sistrip::invalid32_), 
    lldCh_(sistrip::invalid_),
    highMedian_(1.*sistrip::invalid_),
    highMean_(1.*sistrip::invalid_),
    highRms_(1.*sistrip::invalid_),
    lowMedian_(1.*sistrip::invalid_),
    lowMean_(1.*sistrip::invalid_),
    lowRms_(1.*sistrip::invalid_),
    range_(1.*sistrip::invalid_),
    midRange_(1.*sistrip::invalid_),
    max_(1.*sistrip::invalid_),
    min_(1.*sistrip::invalid_),
    histo_(0,"")
{;}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::reset() {
    dcuHardId_ = sistrip::invalid32_; 
    lldCh_ = sistrip::invalid_;
    highMedian_ = 1.*sistrip::invalid_;
    highMean_ = 1.*sistrip::invalid_;
    highRms_ = 1.*sistrip::invalid_;
    lowMedian_ = 1.*sistrip::invalid_;
    lowMean_ = 1.*sistrip::invalid_;
    lowRms_ = 1.*sistrip::invalid_;
    range_ = 1.*sistrip::invalid_;
    midRange_ = 1.*sistrip::invalid_;
    max_ = 1.*sistrip::invalid_;
    min_ = 1.*sistrip::invalid_; 
    histo_ = Histo(0,"");
}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::extract( const std::vector<TH1*>& histos ) { 

  // Check number of histograms
  if ( histos.size() != 1 ) {
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
    if ( title.runType() != sistrip::FAST_CABLING ) {
      addErrorCode(sistrip::unexpectedTask_);
      continue;
    }

    // Extract cabling histo
    histo_.first = *ihis;
    histo_.second = (*ihis)->GetName();
    
  }
  
}

// -----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::analyse() { 


  if ( !histo_.first ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }
  
  TProfile* histo = dynamic_cast<TProfile*>(histo_.first);
  if ( !histo ) {
    addErrorCode(sistrip::nullPtr_);
    return;
  }

  // Initialization
  uint16_t zero_entries = 0;
  uint16_t nbins = static_cast<uint16_t>( histo->GetNbinsX() );
  std::vector<float> contents; 
  std::vector<float> errors;
  std::vector<float> entries;
  contents.reserve( nbins );
  errors.reserve( nbins );
  entries.reserve( nbins );

  // Copy histo contents to containers and find min/max
  max_ = -1.*sistrip::invalid_;
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) {
    contents.push_back( histo->GetBinContent(ibin+1) );
    errors.push_back( histo->GetBinError(ibin+1) );
    entries.push_back( histo->GetBinEntries(ibin+1) );
    if ( entries[ibin] ) { 
      if ( contents[ibin] > max_ ) { max_ = contents[ibin]; }
      if ( contents[ibin] < min_ ) { min_ = contents[ibin]; }
    } else { zero_entries++; }
  }
  if ( max_ < -1. * sistrip::valid_ ) { max_ = sistrip::invalid_; }
  
  // Check number of bins
  if ( contents.size() != nBitsForDcuId_ + nBitsForLldCh_ ) { 
    addErrorCode(sistrip::numberOfBins_);
    return; 
  }
  
  // Check for bins with zero entries
  if ( zero_entries ) { 
    addErrorCode(sistrip::noEntries_);
    return; 
  }

  // Check min and max found
  if ( max_ > sistrip::valid_  || 
       min_ > sistrip::valid_ ) { 
    return; 
  }
  
  // Calculate range and mid-range levels
  range_ = max_ - min_;
  midRange_ = min_ + range_ / 2.;
  
  // Check if range is above threshold
  if ( range_ < threshold_ ) {
    addErrorCode(sistrip::smallDataRange_);
    return; 
  }
  
  // Identify samples to be either "low" or "high"
  std::vector<float> high;
  std::vector<float> low;
  for ( uint16_t ibin = 0; ibin < nbins; ibin++ ) { 
    if ( entries[ibin] ) {
      if ( contents[ibin] < midRange_ ) { 
	low.push_back( contents[ibin] ); 
      } else { 
	high.push_back( contents[ibin] ); 
      }
    }
  }

  // Find median of high and low levels
  sort( high.begin(), high.end() );
  sort( low.begin(), low.end() );
  if ( !high.empty() ) { highMedian_ = high[ high.size()%2 ? high.size()/2 : high.size()/2 ]; }
  if ( !low.empty() ) { lowMedian_ = low[ low.size()%2 ? low.size()/2 : low.size()/2 ]; }

  // Check if light levels above thresholds
  if ( highMedian_ < dirtyThreshold_ ) {
    addErrorCode(sistrip::invalidLightLevel_); 
  }
  if ( lowMedian_ < trimDacThreshold_ ) {
    addErrorCode(sistrip::invalidTrimDacLevel_); 
  }
  
  // Find mean and rms in "low" samples
  lowMean_ = 0.;
  lowRms_ = 0.;
  for ( uint16_t ibin = 0; ibin < low.size(); ibin++ ) {
    lowMean_ += low[ibin];
    lowRms_ += low[ibin] * low[ibin];
  }
  if ( !low.empty() ) { 
    lowMean_ = lowMean_ / low.size();
    lowRms_ = lowRms_ / low.size();
  } else { 
    lowMean_ = 1. * sistrip::invalid_;
    lowRms_ = 1. * sistrip::invalid_;
  }
  if ( lowMean_ < sistrip::valid_ ) { 
    lowRms_ = sqrt( fabs(lowRms_-lowMean_*lowMean_) ); 
  } else {
    lowMean_ = 1. * sistrip::invalid_;
    lowRms_ = 1. * sistrip::invalid_;
  }

  // Find mean and rms in "high" samples
  highMean_ = 0.;
  highRms_ = 0.;
  for ( uint16_t ibin = 0; ibin < high.size(); ibin++ ) {
    highMean_ += high[ibin];
    highRms_ += high[ibin] * high[ibin];
  }
  if ( !high.empty() ) { 
    highMean_ = highMean_ / high.size();
    highRms_ = highRms_ / high.size();
  } else { 
    highMean_ = 1. * sistrip::invalid_;
    highRms_ = 1. * sistrip::invalid_;
  }
  if ( highMean_ < sistrip::valid_ ) { 
    highRms_ = sqrt( fabs(highRms_- highMean_*highMean_) ); 
  } else {
    highMean_ = 1. * sistrip::invalid_;
    highRms_ = 1. * sistrip::invalid_;
  }

  // Check if light levels above thresholds
  if ( highMean_ < dirtyThreshold_ ) {
    addErrorCode(sistrip::invalidLightLevel_); 
  }
  if ( lowMean_ < trimDacThreshold_ ) {
    addErrorCode(sistrip::invalidTrimDacLevel_); 
  }

  // Recalculate range
  if ( highMean_ < 1. * sistrip::valid_ &&
       lowMean_  < 1. * sistrip::valid_ ) { 
    range_ = highMean_ - lowMean_;
    midRange_ = lowMean_ + range_ / 2.;
  } else { 
    range_ = 1. * sistrip::invalid_;
    midRange_ = 1. * sistrip::invalid_;
  }
  
  // Check if updated range is valid and above threshold 
  if ( range_ > 1. * sistrip::valid_ ||
       range_ < threshold_ ) {
    addErrorCode(sistrip::smallDataRange_);
    return; 
  }
  
  // Extract DCU id
  dcuHardId_ = 0;
  for ( uint16_t ibin = 0; ibin < nBitsForDcuId_; ibin++ ) {
    if ( entries[ibin] ) {
      if ( contents[ibin] > midRange_ ) {
	dcuHardId_ += 0xFFFFFFFF & (1<<ibin);
      }
    }
  }
  if ( !dcuHardId_ ) { dcuHardId_ = sistrip::invalid32_; }

  // Extract DCU id
  lldCh_ = 0;
  for ( uint16_t ibin = 0; ibin < nBitsForLldCh_; ibin++ ) {
    if ( entries[nBitsForDcuId_+ibin] ) {
      if ( contents[nBitsForDcuId_+ibin] > midRange_ ) {
	lldCh_ += ( 0x3 & (1<<ibin) );
      }
    }
  }
  lldCh_++; // starts from 1
  if ( !lldCh_ ) { lldCh_ = sistrip::invalid_; }

}

// ----------------------------------------------------------------------------
// 
bool FastFedCablingAnalysis::isValid() const {
  return ( dcuHardId_ < sistrip::invalid32_ &&
	   lldCh_ < sistrip::valid_ && 
	   highMedian_ < sistrip::valid_ && 
	   highMean_ < sistrip::valid_ && 
	   highRms_ < sistrip::valid_ && 
	   lowMedian_ < sistrip::valid_ && 
	   lowMean_ < sistrip::valid_ && 
	   lowRms_ < sistrip::valid_ && 
	   range_ < sistrip::valid_ && 
	   midRange_ < sistrip::valid_ && 
	   max_ < sistrip::valid_ && 
	   min_ < sistrip::valid_ &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
bool FastFedCablingAnalysis::isDirty() const {
  return ( highMean_ < dirtyThreshold_ );
} 

// ----------------------------------------------------------------------------
// 
bool FastFedCablingAnalysis::badTrimDac() const {
  return ( lowMean_ < trimDacThreshold_ );
} 

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::header( std::stringstream& ss ) const { 
  ss << "[" << myName() << "] Monitorables (65535 means \"invalid\"):" << std::endl;

  //summary(ss);

  SiStripFecKey fec_key( fecKey() );
  if ( fec_key.isValid() ) {
    ss << " Crate/FEC/Ring/CCU/Mod/LLD     : " 
       << fec_key.fecCrate() << "/" 
       << fec_key.fecSlot() << "/" 
       << fec_key.fecRing() << "/" 
       << fec_key.ccuAddr() << "/" 
       << fec_key.ccuChan() << "/" 
       << fec_key.lldChan() 
       << std::endl;
  } else {
    ss << " Crate/FEC/Ring/CCU/Mod/LLD     : (invalid)" 
       << std::endl;
  }

  SiStripFedKey fed_key( fedKey() );
  ss << " FedId/FeUnit/FeChan/FedChannel : " 
     << fed_key.fedId() << "/" 
     << fed_key.feUnit() << "/" 
     << fed_key.feChan() << "/"
     << fed_key.fedChannel()
     << std::endl;
  // if ( fed_key.fedChannel() != sistrip::invalid_ ) { ss << fed_key.fedChannel(); }
  // else { ss << "(invalid)"; }
  // ss << std::endl;
  
  ss << " FecKey/Fedkey (hex)            : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << fecKey()
     << " / 0x" 
     << std::setw(8) << std::setfill('0') << fedKey()
     << std::dec
     << std::endl;
  
  ss << " DcuId (hex/dec)                : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << dcuId() 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << dcuId() 
     << std::endl;

  ss << " DetId (hex/dec)                : 0x" 
     << std::hex 
     << std::setw(8) << std::setfill('0') << detId() 
     << " / "
     << std::dec
     << std::setw(10) << std::setfill(' ') << detId() 
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::summary( std::stringstream& ss ) const { 

  SiStripFecKey fec_key( fecKey() );
  SiStripFedKey fed_key( fedKey() );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );
  std::string title = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					 type,
					 sistrip::FED_KEY, 
					 fed_key.key(),
					 sistrip::LLD_CHAN, 
					 fec_key.lldChan() ).title();
  
  ss << " Summary"
     << ":"
     << ( isValid() ? "Valid" : "Invalid" )
     << ":"
     << sistrip::readoutView_ << ":"
     << fed_key.fedId() << "/" 
     << fed_key.feUnit() << "/" 
     << fed_key.feChan() 
     << ":"
     << sistrip::dqmRoot_ << sistrip::dir_ 
     << "Collate" << sistrip::dir_ 
     << fed_key.path()
     << ":"
     << title
     << std::endl;

}

// ----------------------------------------------------------------------------
// 
void FastFedCablingAnalysis::print( std::stringstream& ss, uint32_t not_used ) { 
  header( ss );
  ss <<  std::fixed << std::setprecision(2)
     << " DCU id extracted from histo     : 0x" 
     << std::hex
     << std::setw(8) << std::setfill('0') << dcuHardId_ << std::endl
     << std::dec
     << " LLD chan extracted from histo   : " << lldCh_ << std::endl
     << " \"High\" level (mean+/-rms) [ADC] : " << highMean_ << " +/- " << highRms_ << std::endl
     << " \"Low\" level (mean+/-rms)  [ADC] : " << lowMean_ << " +/- " << lowRms_ << std::endl
     << " Median \"high\" level       [ADC] : " << highMedian_ << std::endl
     << " Median \"low\" level        [ADC] : " << lowMedian_ << std::endl
     << " Range                     [ADC] : " << range_ << std::endl
     << " Mid-range level           [ADC] : " << midRange_ << std::endl
     << " Maximum level             [ADC] : " << max_ << std::endl
     << " Minimum level             [ADC] : " << min_ << std::endl;
  ss << std::boolalpha
     << " isDirty                         : " << isDirty()  << std::endl
     << " isValid                         : " << isValid()  << std::endl
     << std::noboolalpha
     << " Error codes (found "  
     << std::setw(2) << std::setfill(' ') << getErrorCodes().size() 
     << ")          : ";
  if ( getErrorCodes().empty() ) { ss << "(none)"; }
  else { 
    VString::const_iterator istr = getErrorCodes().begin();
    VString::const_iterator jstr = getErrorCodes().end();
    for ( ; istr != jstr; ++istr ) { ss << *istr << " "; }
  }
  ss << std::endl;
}
