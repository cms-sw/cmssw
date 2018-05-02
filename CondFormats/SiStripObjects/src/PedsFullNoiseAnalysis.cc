#include "CondFormats/SiStripObjects/interface/PedsFullNoiseAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace sistrip;


// ----------------------------------------------------------------------------
// 
PedsFullNoiseAnalysis::PedsFullNoiseAnalysis( const uint32_t& key ) 
  : CommissioningAnalysis(key,"PedsFullNoiseAnalysis"),
    peds_(2,VFloat(128,sistrip::invalid_)), 
    noise_(2,VFloat(128,sistrip::invalid_)), 
    raw_(2,VFloat(128,sistrip::invalid_)),
    adProbab_(2,VFloat(128,sistrip::invalid_)),
    ksProbab_(2,VFloat(128,sistrip::invalid_)),
    jbProbab_(2,VFloat(128,sistrip::invalid_)),
    chi2Probab_(2,VFloat(128,sistrip::invalid_)),
    residualRMS_(2,VFloat(128,sistrip::invalid_)),
    residualSigmaGaus_(2,VFloat(128,sistrip::invalid_)),
    noiseSignificance_(2,VFloat(128,sistrip::invalid_)),
    residualMean_(2,VFloat(128,sistrip::invalid_)),
    residualSkewness_(2,VFloat(128,sistrip::invalid_)),
    residualKurtosis_(2,VFloat(128,sistrip::invalid_)),
    residualIntegralNsigma_(2,VFloat(128,sistrip::invalid_)),
    residualIntegral_(2,VFloat(128,sistrip::invalid_)),
    badStripBit_(2,VInt(128,sistrip::invalid_)),
    deadStripBit_(2,VInt(128,sistrip::invalid_)),
    deadStrip_(2,VInt(0,sistrip::invalid_)),
    badStrip_(2,VInt(0,sistrip::invalid_)),
    shiftedStrip_(2,VInt(0,sistrip::invalid_)),
    lowNoiseStrip_(2,VInt(0,sistrip::invalid_)),
    largeNoiseStrip_(2,VInt(0,sistrip::invalid_)),
    largeNoiseSignificance_(2,VInt(0,sistrip::invalid_)),
    badFitStatus_(2,VInt(0,sistrip::invalid_)),
    badADProbab_(2,VInt(0,sistrip::invalid_)),
    badKSProbab_(2,VInt(0,sistrip::invalid_)),
    badJBProbab_(2,VInt(0,sistrip::invalid_)),
    badChi2Probab_(2,VInt(0,sistrip::invalid_)),
    badTailStrip_(2,VInt(0,sistrip::invalid_)),
    badDoublePeakStrip_(2,VInt(0,sistrip::invalid_)),
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    noiseMean_(2,sistrip::invalid_), 
    noiseSpread_(2,sistrip::invalid_), 
    rawMean_(2,sistrip::invalid_), 
    rawSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    noiseMax_(2,sistrip::invalid_), 
    noiseMin_(2,sistrip::invalid_),
    rawMax_(2,sistrip::invalid_), 
    rawMin_(2,sistrip::invalid_),
    legacy_(false)
{
  // for flag bits one reserve at max 128 positions since will be filled with strip id only if the strip is bad
  for(auto iapv : deadStrip_)
    iapv.reserve(128);
  for(auto iapv : badStrip_)
    iapv.reserve(128);
  for(auto iapv : shiftedStrip_)
    iapv.reserve(128);
  for(auto iapv : lowNoiseStrip_)
    iapv.reserve(128);
  for(auto iapv : largeNoiseStrip_)
    iapv.reserve(128);
  for(auto iapv : largeNoiseSignificance_)
    iapv.reserve(128);
  for(auto iapv : badFitStatus_)
    iapv.reserve(128);
  for(auto iapv : badADProbab_)
    iapv.reserve(128);
  for(auto iapv : badKSProbab_)
    iapv.reserve(128);
  for(auto iapv : badJBProbab_)
    iapv.reserve(128);
  for(auto iapv : badChi2Probab_)
    iapv.reserve(128);
  for(auto iapv : badTailStrip_)
    iapv.reserve(128);
  for(auto iapv : badDoublePeakStrip_)
    iapv.reserve(128);  
}

// ----------------------------------------------------------------------------
// 
PedsFullNoiseAnalysis::PedsFullNoiseAnalysis() 
  : CommissioningAnalysis("PedsFullNoiseAnalysis"),
    peds_(2,VFloat(128,sistrip::invalid_)), 
    noise_(2,VFloat(128,sistrip::invalid_)), 
    raw_(2,VFloat(128,sistrip::invalid_)),
    adProbab_(2,VFloat(128,sistrip::invalid_)),
    ksProbab_(2,VFloat(128,sistrip::invalid_)),
    jbProbab_(2,VFloat(128,sistrip::invalid_)),
    chi2Probab_(2,VFloat(128,sistrip::invalid_)),
    residualRMS_(2,VFloat(128,sistrip::invalid_)),
    residualSigmaGaus_(2,VFloat(128,sistrip::invalid_)),
    noiseSignificance_(2,VFloat(128,sistrip::invalid_)),
    residualMean_(2,VFloat(128,sistrip::invalid_)),
    residualSkewness_(2,VFloat(128,sistrip::invalid_)),
    residualKurtosis_(2,VFloat(128,sistrip::invalid_)),
    residualIntegralNsigma_(2,VFloat(128,sistrip::invalid_)),
    residualIntegral_(2,VFloat(128,sistrip::invalid_)),
    badStripBit_(2,VInt(128,sistrip::invalid_)),
    deadStripBit_(2,VInt(128,sistrip::invalid_)),
    deadStrip_(2,VInt(0,sistrip::invalid_)),
    badStrip_(2,VInt(0,sistrip::invalid_)),
    shiftedStrip_(2,VInt(0,sistrip::invalid_)),
    lowNoiseStrip_(2,VInt(0,sistrip::invalid_)),
    largeNoiseStrip_(2,VInt(0,sistrip::invalid_)),
    largeNoiseSignificance_(2,VInt(0,sistrip::invalid_)),
    badFitStatus_(2,VInt(0,sistrip::invalid_)),
    badADProbab_(2,VInt(0,sistrip::invalid_)),
    badKSProbab_(2,VInt(0,sistrip::invalid_)),
    badJBProbab_(2,VInt(0,sistrip::invalid_)),
    badChi2Probab_(2,VInt(0,sistrip::invalid_)),
    badTailStrip_(2,VInt(0,sistrip::invalid_)),
    badDoublePeakStrip_(2,VInt(128,sistrip::invalid_)),
    pedsMean_(2,sistrip::invalid_), 
    pedsSpread_(2,sistrip::invalid_), 
    noiseMean_(2,sistrip::invalid_), 
    noiseSpread_(2,sistrip::invalid_), 
    rawMean_(2,sistrip::invalid_), 
    rawSpread_(2,sistrip::invalid_), 
    pedsMax_(2,sistrip::invalid_), 
    pedsMin_(2,sistrip::invalid_), 
    noiseMax_(2,sistrip::invalid_), 
    noiseMin_(2,sistrip::invalid_),
    rawMax_(2,sistrip::invalid_), 
    rawMin_(2,sistrip::invalid_),
    legacy_(false)
{
  // for flag bits one reserve at max 128 positions since will be filled with strip id only if the strip is bad
  for(auto iapv : deadStrip_)
    iapv.reserve(128);
  for(auto iapv : badStrip_)
    iapv.reserve(128);
  for(auto iapv : shiftedStrip_)
    iapv.reserve(128);
  for(auto iapv : lowNoiseStrip_)
    iapv.reserve(128);
  for(auto iapv : largeNoiseStrip_)
    iapv.reserve(128);
  for(auto iapv : largeNoiseSignificance_)
    iapv.reserve(128);
  for(auto iapv : badFitStatus_)
    iapv.reserve(128);
  for(auto iapv : badADProbab_)
    iapv.reserve(128);
  for(auto iapv : badKSProbab_)
    iapv.reserve(128);
  for(auto iapv : badJBProbab_)
    iapv.reserve(128);
  for(auto iapv : badChi2Probab_)
    iapv.reserve(128);
  for(auto iapv : badTailStrip_)
    iapv.reserve(128);
  for(auto iapv : badDoublePeakStrip_)
    iapv.reserve(128);  
}

// ----------------------------------------------------------------------------
// 
void PedsFullNoiseAnalysis::reset() {

  peds_     = VVFloat(2,VFloat(128,sistrip::invalid_));
  noise_    = VVFloat(2,VFloat(128,sistrip::invalid_));
  raw_      = VVFloat(2,VFloat(128,sistrip::invalid_));
  adProbab_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  ksProbab_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  jbProbab_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  chi2Probab_  = VVFloat(2,VFloat(128,sistrip::invalid_));
  residualRMS_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  residualSigmaGaus_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  noiseSignificance_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  residualMean_     = VVFloat(2,VFloat(128,sistrip::invalid_));
  residualSkewness_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  residualKurtosis_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  residualIntegralNsigma_ = VVFloat(2,VFloat(128,sistrip::invalid_));
  residualIntegral_       = VVFloat(2,VFloat(128,sistrip::invalid_));

  deadStrip_     = VVInt(2,VInt(0,sistrip::invalid_)); 
  badStrip_      = VVInt(2,VInt(0,sistrip::invalid_)); 
  badStripBit_   = VVInt(2,VInt(128,sistrip::invalid_)); 
  deadStripBit_  = VVInt(2,VInt(128,sistrip::invalid_)); 
  shiftedStrip_  = VVInt(2,VInt(0,sistrip::invalid_)); 
  lowNoiseStrip_ = VVInt(2,VInt(0,sistrip::invalid_));  
  largeNoiseStrip_ = VVInt(2,VInt(0,sistrip::invalid_));
  largeNoiseSignificance_ = VVInt(2,VInt(0,sistrip::invalid_));
  badFitStatus_ = VVInt(2,VInt(0,sistrip::invalid_));
  badADProbab_  = VVInt(2,VInt(0,sistrip::invalid_));
  badKSProbab_  = VVInt(2,VInt(0,sistrip::invalid_));
  badJBProbab_  = VVInt(2,VInt(0,sistrip::invalid_));
  badChi2Probab_ = VVInt(2,VInt(0,sistrip::invalid_));
  badTailStrip_ = VVInt(2,VInt(0,sistrip::invalid_));
  badDoublePeakStrip_ = VVInt(2,VInt(0,sistrip::invalid_));
        
  // for flag bits one reserve at max 128 positions since will be filled with strip id only if the strip is bad
  for(auto iapv : deadStrip_)
    iapv.reserve(128);
  for(auto iapv : badStrip_)
    iapv.reserve(128);
  for(auto iapv : shiftedStrip_)
    iapv.reserve(128);
  for(auto iapv : lowNoiseStrip_)
    iapv.reserve(128);
  for(auto iapv : largeNoiseStrip_)
    iapv.reserve(128);
  for(auto iapv : largeNoiseSignificance_)
    iapv.reserve(128);
  for(auto iapv : badFitStatus_)
    iapv.reserve(128);
  for(auto iapv : badADProbab_)
    iapv.reserve(128);
  for(auto iapv : badKSProbab_)
    iapv.reserve(128);
  for(auto iapv : badJBProbab_)
    iapv.reserve(128);
  for(auto iapv : badChi2Probab_)
    iapv.reserve(128);
  for(auto iapv : badTailStrip_)
    iapv.reserve(128);
  for(auto iapv : badDoublePeakStrip_)
    iapv.reserve(128);  

  pedsMean_    = VFloat(2,sistrip::invalid_); 
  pedsSpread_  = VFloat(2,sistrip::invalid_); 
  noiseMean_   = VFloat(2,sistrip::invalid_); 
  noiseSpread_ = VFloat(2,sistrip::invalid_); 
  rawMean_     = VFloat(2,sistrip::invalid_);
  rawSpread_   = VFloat(2,sistrip::invalid_);
  pedsMax_     = VFloat(2,sistrip::invalid_); 
  pedsMin_     = VFloat(2,sistrip::invalid_); 
  noiseMax_    = VFloat(2,sistrip::invalid_); 
  noiseMin_    = VFloat(2,sistrip::invalid_);
  rawMax_      = VFloat(2,sistrip::invalid_);
  rawMin_      = VFloat(2,sistrip::invalid_);

  legacy_ = false;
}

// ----------------------------------------------------------------------------
// 
bool PedsFullNoiseAnalysis::isValid() const {
  return ( pedsMean_[0] < sistrip::maximum_ &&
	   pedsMean_[1] < sistrip::maximum_ &&
	   pedsSpread_[0] < sistrip::maximum_ &&
	   pedsSpread_[1] < sistrip::maximum_ &&
	   noiseMean_[0] < sistrip::maximum_ &&
	   noiseMean_[1] < sistrip::maximum_ &&
	   noiseSpread_[0] < sistrip::maximum_ &&
	   noiseSpread_[1] < sistrip::maximum_ &&
	   rawMean_[0] < sistrip::maximum_ &&
	   rawMean_[1] < sistrip::maximum_ &&
	   rawSpread_[0] < sistrip::maximum_ &&
	   rawSpread_[1] < sistrip::maximum_ &&
	   pedsMax_[0] < sistrip::maximum_ &&
	   pedsMax_[1] < sistrip::maximum_ &&
	   pedsMin_[0] < sistrip::maximum_ &&
	   pedsMin_[1] < sistrip::maximum_ &&
	   noiseMax_[0] < sistrip::maximum_ &&
	   noiseMax_[1] < sistrip::maximum_ &&
	   noiseMin_[0] < sistrip::maximum_ &&
	   noiseMin_[1] < sistrip::maximum_ &&
	   rawMax_[0] < sistrip::maximum_ &&
	   rawMax_[1] < sistrip::maximum_ &&
	   rawMin_[0] < sistrip::maximum_ &&
	   rawMin_[1] < sistrip::maximum_ &&
	   getErrorCodes().empty() );
} 

// ----------------------------------------------------------------------------
// 
void PedsFullNoiseAnalysis::summary( std::stringstream& ss ) const { 

  SiStripFecKey fec_key( fecKey() );
  SiStripFedKey fed_key( fedKey() );
  
  sistrip::RunType type = SiStripEnumsAndStrings::runType( myName() );

  std::stringstream extra1,extra2,extra3;
  if ( legacy_ ) {
    extra1 << sistrip::extrainfo::pedsAndRawNoise_; 
    extra2 << sistrip::extrainfo::pedsAndCmSubNoise_; 
    extra3 << sistrip::extrainfo::commonMode_;
  } else {
    extra1 << sistrip::extrainfo::pedestals_; 
    extra2 << sistrip::extrainfo::rawNoise_; 
    extra3 << sistrip::extrainfo::commonMode_;
  }
  
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
					  sistrip::APV, 
					  SiStripFecKey::i2cAddr( fec_key.lldChan(), true ),
					  extra3.str() ).title();
  std::string title4 = SiStripHistoTitle( sistrip::EXPERT_HISTO, 
					  type,
					  sistrip::FED_KEY, 
					  fed_key.key(),
					  sistrip::APV, 
					  SiStripFecKey::i2cAddr( fec_key.lldChan(), false ),
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
     << title1 << ";" << title2 << ";" << title3 << ";" << title4
     << std::endl;
  
}

// ----------------------------------------------------------------------------
// 
void PedsFullNoiseAnalysis::print( std::stringstream& ss, uint32_t iapv ) { 

  if ( iapv == 1 || iapv == 2 ) { iapv--; }
  else { iapv = 0; }
  
  if ( peds_[iapv].size()  < 128 ||
       noise_[iapv].size() < 128 ||
       raw_[iapv].size()   < 128 ) { 

    edm::LogWarning(mlCommissioning_)
      << "[" << myName() << "::" << __func__ << "]"
      << " Unexpected number of pedestal/noise values: " 
      << peds_[iapv].size() << ", " 
      << noise_[iapv].size() << ", " 
      << raw_[iapv].size();
    return;
  }
  
  header( ss );
  ss << " Monitorables for APV number     : " << iapv;
  if ( iapv == 0 ) { ss << " (first of pair)"; }
  else if ( iapv == 1 ) { ss << " (second of pair)"; } 
  ss << std::endl;
  ss << std::fixed << std::setprecision(2);
  ss << " Example peds/noise for strips   : "
     << "     0,     31,     63,    127" << std::endl
     << "  Peds                     [ADC] : " 
     << std::setw(6) << peds_[iapv][0] << ", " 
     << std::setw(6) << peds_[iapv][31] << ", " 
     << std::setw(6) << peds_[iapv][63] << ", " 
     << std::setw(6) << peds_[iapv][127] << std::endl
     << "  Noise                    [ADC] : " 
     << std::setw(6) << noise_[iapv][0] << ", " 
     << std::setw(6) << noise_[iapv][31] << ", " 
     << std::setw(6) << noise_[iapv][63] << ", " 
     << std::setw(6) << noise_[iapv][127] << std::endl
     << "  Raw noise                [ADC] : " 
     << std::setw(6) << raw_[iapv][0] << ", " 
     << std::setw(6) << raw_[iapv][31] << ", " 
     << std::setw(6) << raw_[iapv][63] << ", " 
     << std::setw(6) << raw_[iapv][127] << std::endl
     << " Dead strips (<5s)       [strip] : (" << deadStrip_[iapv].size() << " in total) ";
  
  for ( uint16_t ii = 0; ii < deadStrip_[iapv].size(); ii++ ) { 
    ss << deadStrip_[iapv][ii] << " "; }
  
  ss << std::endl;
  ss << " Bad strips (>5s)      [strip] : (" << badStrip_[iapv].size() << " in total) ";
  for ( uint16_t ii = 0; ii < badStrip_[iapv].size(); ii++ ) { 
    ss << badStrip_[iapv][ii] << " "; 
  } 
  ss << std::endl;
  ss << " Mean peds +/- spread      [ADC] : " << pedsMean_[iapv] << " +/- " << pedsSpread_[iapv] << std::endl 
     << " Min/Max pedestal          [ADC] : " << pedsMin_[iapv] << " <-> " << pedsMax_[iapv] << std::endl
     << " Mean noise +/- spread     [ADC] : " << noiseMean_[iapv] << " +/- " << noiseSpread_[iapv] << std::endl 
     << " Min/Max noise             [ADC] : " << noiseMin_[iapv] << " <-> " << noiseMax_[iapv] << std::endl
     << " Mean raw noise +/- spread [ADC] : " << rawMean_[iapv] << " +/- " << rawSpread_[iapv] << std::endl 
     << " Min/Max raw noise         [ADC] : " << rawMin_[iapv] << " <-> " << rawMax_[iapv] << std::endl
     << " Normalised noise                : " << "(yet to be implemented...)" << std::endl
     << std::boolalpha 
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

