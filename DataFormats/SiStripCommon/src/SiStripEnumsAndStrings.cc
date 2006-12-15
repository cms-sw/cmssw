#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
string SiStripHistoNamingScheme::view( const sistrip::View& view ) {
  if      ( view == sistrip::READOUT ) { return sistrip::readoutView_; }
  else if ( view == sistrip::CONTROL ) { return sistrip::controlView_; }
  else if ( view == sistrip::DETECTOR ) { return sistrip::detectorView_; }
  else if ( view == sistrip::UNDEFINED_VIEW ) { return sistrip::undefinedView_; }
  else { return sistrip::unknownView_; }
}

// -----------------------------------------------------------------------------
//
sistrip::View SiStripHistoNamingScheme::view( const string& dir ) {
  if      ( dir.find( sistrip::readoutView_ ) != string::npos ) { return sistrip::READOUT; } 
  else if ( dir.find( sistrip::controlView_ ) != string::npos ) { return sistrip::CONTROL; } 
  else if ( dir.find( sistrip::detectorView_ ) != string::npos ) { return sistrip::DETECTOR; } 
  else if ( dir.find( sistrip::undefinedView_ ) != string::npos ) { return sistrip::UNDEFINED_VIEW; } 
  else { return sistrip::UNKNOWN_VIEW; }
}

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::task( const sistrip::Task& task ) {
  if ( task == sistrip::FED_CABLING ) { return sistrip::fedCabling_; }
  else if ( task == sistrip::APV_TIMING ) { return sistrip::apvTiming_; }
  else if ( task == sistrip::FED_TIMING ) { return sistrip::fedTiming_; }
  else if ( task == sistrip::OPTO_SCAN ) { return sistrip::optoScan_; }
  else if ( task == sistrip::VPSP_SCAN ) { return sistrip::vpspScan_; }
  else if ( task == sistrip::PEDESTALS ) { return sistrip::pedestals_; }
  else if ( task == sistrip::APV_LATENCY ){ return sistrip::apvLatency_; }
  else if ( task == sistrip::DAQ_SCOPE_MODE ){ return sistrip::daqScopeMode_; }
  else if ( task == sistrip::PHYSICS ){ return sistrip::physics_; }
  else if ( task == sistrip::UNDEFINED_TASK ) { return sistrip::undefinedTask_; }
  else { return sistrip::unknownTask_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::Task SiStripHistoNamingScheme::task( const string& task ) {
  if ( task == "FED_CABLING" || 
       task.find( sistrip::fedCabling_ ) != string::npos ) { return sistrip::FED_CABLING; }
  else if ( task == "APV_TIMING" || 
	    task.find( sistrip::apvTiming_ ) != string::npos ) { return sistrip::APV_TIMING; }
  else if ( task == "FED_TIMING" || 
	    task.find( sistrip::fedTiming_ ) != string::npos ) { return sistrip::FED_TIMING; }
  else if ( task == "OPTO_SCAN" || 
	    task.find( sistrip::optoScan_ ) != string::npos ) { return sistrip::OPTO_SCAN; }
  else if ( task == "VPSP_SCAN" || 
	    task.find( sistrip::vpspScan_ ) != string::npos ) { return sistrip::VPSP_SCAN; }
  else if ( task == "PEDESTALS" || 
	    task.find( sistrip::pedestals_ ) != string::npos ) { return sistrip::PEDESTALS; }
  else if ( task == "APV_LATENCY" || 
	    task.find( sistrip::apvLatency_ ) != string::npos ) { return sistrip::APV_LATENCY; }
  else if ( task == "DAQ_SCOPE_MODE" || 
	    task.find( sistrip::daqScopeMode_ ) != string::npos ) { return sistrip::DAQ_SCOPE_MODE; }
  else if ( task == "PHYSICS" || 
	    task.find( sistrip::physics_ ) != string::npos ) { return sistrip::PHYSICS; }
  else if ( task == "UNDEFINED" || 
	    task.find( sistrip::undefinedTask_ ) != string::npos ) { return sistrip::UNDEFINED_TASK; }
  else { return sistrip::UNKNOWN_TASK; }
}

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::keyType( const sistrip::KeyType& key_type ) {
  if ( key_type == sistrip::FED_KEY ) { return sistrip::fedKey_; }
  else if ( key_type == sistrip::FEC_KEY ) { return sistrip::fecKey_; }
  else if ( key_type == sistrip::DET_KEY ) { return sistrip::detKey_; }
  //else if ( key_type == sistrip::NO_KEY ) { return ""; }
  else if ( key_type == sistrip::UNDEFINED_KEY )  { return sistrip::undefinedKey_; }
  else { return sistrip::unknownKey_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::KeyType SiStripHistoNamingScheme::keyType( const string& key_type ) {
  if ( key_type.find ( sistrip::fedKey_) != string::npos ) { return sistrip::FED_KEY; }
  else if ( key_type.find ( sistrip::fecKey_) != string::npos ) { return sistrip::FEC_KEY; }
  else if ( key_type.find ( sistrip::detKey_) != string::npos ) { return sistrip::DET_KEY; }
  //else if ( key_type == "" ) { return sistrip::NO_KEY; }
  else if ( key_type.find ( sistrip::undefinedKey_) != string::npos ) { return sistrip::UNDEFINED_KEY; }
  else { return sistrip::UNKNOWN_KEY; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::granularity( const sistrip::Granularity& granularity ) {
  // System
  if ( granularity == sistrip::TRACKER ) { return sistrip::tracker_; }
  else if ( granularity == sistrip::PARTITION ) { return sistrip::partition_; }
  else if ( granularity == sistrip::TIB ) { return sistrip::tib_; }
  else if ( granularity == sistrip::TOB ) { return sistrip::tob_; }
  else if ( granularity == sistrip::TEC ) { return sistrip::tec_; }
  // Control
  else if ( granularity == sistrip::FEC_CRATE ) { return sistrip::fecCrate_; }
  else if ( granularity == sistrip::FEC_SLOT ) { return sistrip::fecSlot_; }
  else if ( granularity == sistrip::FEC_RING ) { return sistrip::fecRing_; }
  else if ( granularity == sistrip::CCU_ADDR ) { return sistrip::ccuAddr_; }
  else if ( granularity == sistrip::CCU_CHAN ) { return sistrip::ccuChan_; }
  // Readout
  else if ( granularity == sistrip::FED ) { return sistrip::fedId_; }
  else if ( granularity == sistrip::FED_CHANNEL ) { return sistrip::fedChannel_; }
  else if ( granularity == sistrip::FE_UNIT ) { return sistrip::feUnit_; }
  else if ( granularity == sistrip::FE_CHAN ) { return sistrip::feChan_; }
  // Sub-structure
  else if ( granularity == sistrip::LAYER ) { return sistrip::layer_; }
  else if ( granularity == sistrip::ROD ) { return sistrip::rod_; }
  else if ( granularity == sistrip::STRING ) { return sistrip::string_; }
  else if ( granularity == sistrip::DISK ) { return sistrip::disk_; }
  else if ( granularity == sistrip::PETAL ) { return sistrip::petal_; }
  else if ( granularity == sistrip::RING ) { return sistrip::ring_; }
  // Module and below
  else if ( granularity == sistrip::MODULE ) { return sistrip::module_; }
  else if ( granularity == sistrip::LLD_CHAN ) { return sistrip::lldChan_; }
  else if ( granularity == sistrip::APV ) { return sistrip::apv_; }
  // Unknown
  //else if ( granularity == sistrip::NO_GRAN ) { return ""; }
  else if ( granularity == sistrip::UNDEFINED_GRAN ) { return sistrip::undefinedGranularity_; }
  else { return sistrip::unknownGranularity_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::Granularity SiStripHistoNamingScheme::granularity( const string& granularity ) {
  // System
  if ( granularity.find( sistrip::tracker_ ) != string::npos ) { return sistrip::TRACKER; }
  else if ( granularity.find( sistrip::partition_ ) != string::npos ) { return sistrip::PARTITION; }
  else if ( granularity.find( sistrip::tib_ ) != string::npos ) { return sistrip::TIB; }
  else if ( granularity.find( sistrip::tob_ ) != string::npos ) { return sistrip::TOB; }
  else if ( granularity.find( sistrip::tec_ ) != string::npos ) { return sistrip::TEC; }
  // Control
  else if ( granularity.find( sistrip::fecCrate_ ) != string::npos ) { return sistrip::FEC_CRATE; }
  else if ( granularity.find( sistrip::fecSlot_ ) != string::npos ) { return sistrip::FEC_SLOT; }
  else if ( granularity.find( sistrip::fecRing_ ) != string::npos ) { return sistrip::FEC_RING; }
  else if ( granularity.find( sistrip::ccuAddr_ ) != string::npos ) { return sistrip::CCU_ADDR; }
  else if ( granularity.find( sistrip::ccuChan_ ) != string::npos ) { return sistrip::CCU_CHAN; }
  // Readout
  else if ( granularity.find( sistrip::fedId_ ) != string::npos ) { return sistrip::FED; }
  else if ( granularity.find( sistrip::fedChannel_ ) != string::npos ) { return sistrip::FED_CHANNEL; }
  else if ( granularity.find( sistrip::feUnit_ ) != string::npos ) { return sistrip::FE_UNIT; }
  else if ( granularity.find( sistrip::feChan_ ) != string::npos ) { return sistrip::FE_CHAN; }
  // Sub-structure
  else if ( granularity.find( sistrip::layer_ ) != string::npos ) { return sistrip::LAYER; }
  else if ( granularity.find( sistrip::rod_ ) != string::npos ) { return sistrip::ROD; }
  else if ( granularity.find( sistrip::string_ ) != string::npos ) { return sistrip::STRING; }
  else if ( granularity.find( sistrip::disk_ ) != string::npos ) { return sistrip::DISK; }
  else if ( granularity.find( sistrip::petal_ ) != string::npos ) { return sistrip::PETAL; }
  else if ( granularity.find( sistrip::ring_ ) != string::npos ) { return sistrip::RING; }
  // Module and below
  else if ( granularity.find( sistrip::module_ ) != string::npos ) { return sistrip::MODULE; }
  else if ( granularity.find( sistrip::lldChan_ ) != string::npos ) { return sistrip::LLD_CHAN; }
  else if ( granularity.find( sistrip::apv_ ) != string::npos ) { return sistrip::APV; }
  // Unknown
  //else if ( granularity == "" ) { return sistrip::NO_GRAN; }
  else if ( granularity.find( sistrip::undefinedGranularity_ ) != string::npos ) { return sistrip::UNDEFINED_GRAN; }
  else { return sistrip::UNKNOWN_GRAN; }
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::monitorable( const sistrip::Monitorable& mon ) {
  
  // fed cabling
  if ( mon == sistrip::FED_CABLING_FED_ID ) { return sistrip::fedCablingFedId_; } 
  else if ( mon == sistrip::FED_CABLING_FED_CH ) { return sistrip::fedCablingFedCh_; } 
  else if ( mon == sistrip::FED_CABLING_ADC_LEVEL ) { return sistrip::fedCablingAdcLevel_; }
  
  // apv timing
  else if ( mon == sistrip::APV_TIMING_TIME ) { return sistrip::apvTimingTime_; } 
  else if ( mon == sistrip::APV_TIMING_MAX_TIME ) { return sistrip::apvTimingMax_; }
  else if ( mon == sistrip::APV_TIMING_DELAY ) { return sistrip::apvTimingDelay_; }
  else if ( mon == sistrip::APV_TIMING_ERROR ) { return sistrip::apvTimingError_; }
  else if ( mon == sistrip::APV_TIMING_BASE ) { return sistrip::apvTimingBase_; }
  else if ( mon == sistrip::APV_TIMING_PEAK ) { return sistrip::apvTimingPeak_; }
  else if ( mon == sistrip::APV_TIMING_HEIGHT ) { return sistrip::apvTimingHeight_; }

  // fed timing
  else if ( mon == sistrip::FED_TIMING_TIME ) { return sistrip::fedTimingTime_; } 
  else if ( mon == sistrip::FED_TIMING_MAX_TIME ) { return sistrip::fedTimingMax_; }
  else if ( mon == sistrip::FED_TIMING_DELAY ) { return sistrip::fedTimingDelay_; }
  else if ( mon == sistrip::FED_TIMING_ERROR ) { return sistrip::fedTimingError_; }
  else if ( mon == sistrip::FED_TIMING_BASE ) { return sistrip::fedTimingBase_; }
  else if ( mon == sistrip::FED_TIMING_PEAK ) { return sistrip::fedTimingPeak_; }
  else if ( mon == sistrip::FED_TIMING_HEIGHT ) { return sistrip::fedTimingHeight_; }

  // opto scan
  else if ( mon == sistrip::OPTO_SCAN_LLD_GAIN_SETTING ) { return sistrip::optoScanLldGain_; }
  else if ( mon == sistrip::OPTO_SCAN_LLD_BIAS_SETTING ) { return sistrip::optoScanLldBias_; }
  else if ( mon == sistrip::OPTO_SCAN_MEASURED_GAIN ) { return sistrip::optoScanMeasGain_; }
  else if ( mon == sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL ) { return sistrip::optoScanZeroLight_; }
  else if ( mon == sistrip::OPTO_SCAN_LINK_NOISE ) { return sistrip::optoScanLinkNoise_; }
  else if ( mon == sistrip::OPTO_SCAN_BASELINE_LIFT_OFF ) { return sistrip::optoScanBaseLiftOff_; }
  else if ( mon == sistrip::OPTO_SCAN_LASER_THRESHOLD ) { return sistrip::optoScanLaserThresh_; }
  else if ( mon == sistrip::OPTO_SCAN_TICK_HEIGHT ) { return sistrip::optoScanTickHeight_; }

  // vpsp scan
  else if ( mon == sistrip::VPSP_SCAN_BOTH_APVS ) { return sistrip::vpspScanBothApvs_; }
  else if ( mon == sistrip::VPSP_SCAN_APV0 ) { return sistrip::vpspScanApv0_; }
  else if ( mon == sistrip::VPSP_SCAN_APV1 ) { return sistrip::vpspScanApv1_; }

  // pedestals / noise
  else if ( mon == sistrip::PEDESTALS_ALL_STRIPS ) { return sistrip::pedestalsAllStrips_; }
  else if ( mon == sistrip::PEDESTALS_MEAN ) { return sistrip::pedestalsMean_; }
  else if ( mon == sistrip::PEDESTALS_SPREAD ) { return sistrip::pedestalsSpread_; }
  else if ( mon == sistrip::PEDESTALS_MAX ) { return sistrip::pedestalsMax_; }
  else if ( mon == sistrip::PEDESTALS_MIN ) { return sistrip::pedestalsMin_; }
  else if ( mon == sistrip::NOISE_ALL_STRIPS ) { return sistrip::noiseAllStrips_; }
  else if ( mon == sistrip::NOISE_MEAN ) { return sistrip::noiseMean_; }
  else if ( mon == sistrip::NOISE_SPREAD ) { return sistrip::noiseSpread_; }
  else if ( mon == sistrip::NOISE_MAX ) { return sistrip::noiseMax_; }
  else if ( mon == sistrip::NOISE_MIN ) { return sistrip::noiseMin_; }
  else if ( mon == sistrip::NUM_OF_DEAD ) { return sistrip::numOfDead_; }
  else if ( mon == sistrip::NUM_OF_NOISY ) { return sistrip::numOfNoisy_; }

  // scope mode 
  else if ( mon == sistrip::DAQ_SCOPE_MODE_MEAN_SIGNAL ) { return sistrip::daqScopeModeMeanSignal_; }

  // unknown
  else if ( mon == sistrip::UNDEFINED_MONITORABLE ) { return sistrip::undefinedMonitorable_; }
  else { return sistrip::unknownMonitorable_; }
  
}

// -----------------------------------------------------------------------------
// 
sistrip::Monitorable SiStripHistoNamingScheme::monitorable( const string& mon ) {

  // fed cabling
  if ( mon.find( sistrip::fedCablingFedId_ ) != string::npos ) { return sistrip::FED_CABLING_FED_ID; } 
  else if ( mon.find( sistrip::fedCablingFedCh_ ) != string::npos ) { return sistrip::FED_CABLING_FED_CH; } 
  else if ( mon.find( sistrip::fedCablingAdcLevel_ ) != string::npos ) { return sistrip::FED_CABLING_ADC_LEVEL; } 

  // apv timing
  else if ( mon.find( sistrip::apvTimingTime_ ) != string::npos ) { return sistrip::APV_TIMING_TIME; } 
  else if ( mon.find( sistrip::apvTimingMax_ ) != string::npos ) { return sistrip::APV_TIMING_MAX_TIME; }
  else if ( mon.find( sistrip::apvTimingDelay_ ) != string::npos ) { return sistrip::APV_TIMING_DELAY; }
  else if ( mon.find( sistrip::apvTimingError_ ) != string::npos ) { return sistrip::APV_TIMING_ERROR; }
  else if ( mon.find( sistrip::apvTimingBase_ ) != string::npos ) { return sistrip::APV_TIMING_BASE; }
  else if ( mon.find( sistrip::apvTimingPeak_ ) != string::npos ) { return sistrip::APV_TIMING_PEAK; }
  else if ( mon.find( sistrip::apvTimingHeight_ ) != string::npos ) { return sistrip::APV_TIMING_HEIGHT; }

  // fed timing
  else if ( mon.find( sistrip::fedTimingTime_ ) != string::npos ) { return sistrip::FED_TIMING_TIME; } 
  else if ( mon.find( sistrip::fedTimingMax_ ) != string::npos ) { return sistrip::FED_TIMING_MAX_TIME; }
  else if ( mon.find( sistrip::fedTimingDelay_ ) != string::npos ) { return sistrip::FED_TIMING_DELAY; }
  else if ( mon.find( sistrip::fedTimingError_ ) != string::npos ) { return sistrip::FED_TIMING_ERROR; }
  else if ( mon.find( sistrip::fedTimingBase_ ) != string::npos ) { return sistrip::FED_TIMING_BASE; }
  else if ( mon.find( sistrip::fedTimingPeak_ ) != string::npos ) { return sistrip::FED_TIMING_PEAK; }
  else if ( mon.find( sistrip::fedTimingHeight_ ) != string::npos ) { return sistrip::FED_TIMING_HEIGHT; }

  // opto scan
  else if ( mon.find( sistrip::optoScanLldGain_ ) != string::npos ) { return sistrip::OPTO_SCAN_LLD_GAIN_SETTING; }
  else if ( mon.find( sistrip::optoScanLldBias_ ) != string::npos ) { return sistrip::OPTO_SCAN_LLD_BIAS_SETTING; }
  else if ( mon.find( sistrip::optoScanMeasGain_ ) != string::npos ) { return sistrip::OPTO_SCAN_MEASURED_GAIN; }
  else if ( mon.find( sistrip::optoScanZeroLight_ ) != string::npos ) { return sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL; }
  else if ( mon.find( sistrip::optoScanLinkNoise_ ) != string::npos ) { return sistrip::OPTO_SCAN_LINK_NOISE; }
  else if ( mon.find( sistrip::optoScanBaseLiftOff_ ) != string::npos ) { return sistrip::OPTO_SCAN_BASELINE_LIFT_OFF; }
  else if ( mon.find( sistrip::optoScanLaserThresh_ ) != string::npos ) { return sistrip::OPTO_SCAN_LASER_THRESHOLD; }
  else if ( mon.find( sistrip::optoScanTickHeight_ ) != string::npos ) { return sistrip::OPTO_SCAN_TICK_HEIGHT; }

  // vpsp scan
  else if ( mon.find( sistrip::vpspScanBothApvs_ ) != string::npos ) { return sistrip::VPSP_SCAN_BOTH_APVS; }
  else if ( mon.find( sistrip::vpspScanApv0_ ) != string::npos ) { return sistrip::VPSP_SCAN_APV0; }
  else if ( mon.find( sistrip::vpspScanApv1_ ) != string::npos ) { return sistrip::VPSP_SCAN_APV1; }

  // pedestals / noise
  else if ( mon.find( sistrip::pedestalsAllStrips_ ) != string::npos ) { return sistrip::PEDESTALS_ALL_STRIPS; }
  else if ( mon.find( sistrip::pedestalsMean_ ) != string::npos ) { return sistrip::PEDESTALS_MEAN; }
  else if ( mon.find( sistrip::pedestalsSpread_ ) != string::npos ) { return sistrip::PEDESTALS_SPREAD; }
  else if ( mon.find( sistrip::pedestalsMax_ ) != string::npos ) { return sistrip::PEDESTALS_MAX; }
  else if ( mon.find( sistrip::pedestalsMin_ ) != string::npos ) { return sistrip::PEDESTALS_MIN; }
  else if ( mon.find( sistrip::noiseAllStrips_ ) != string::npos ) { return sistrip::NOISE_ALL_STRIPS; }
  else if ( mon.find( sistrip::noiseMean_ ) != string::npos ) { return sistrip::NOISE_MEAN; }
  else if ( mon.find( sistrip::noiseSpread_ ) != string::npos ) { return sistrip::NOISE_SPREAD; }
  else if ( mon.find( sistrip::noiseMax_ ) != string::npos ) { return sistrip::NOISE_MAX; }
  else if ( mon.find( sistrip::noiseMin_ ) != string::npos ) { return sistrip::NOISE_MIN; }
  else if ( mon.find( sistrip::numOfDead_ ) != string::npos ) { return sistrip::NUM_OF_DEAD; }
  else if ( mon.find( sistrip::numOfNoisy_ ) != string::npos ) { return sistrip::NUM_OF_NOISY; }
  
  // scope mode
  else if ( mon.find( sistrip::daqScopeModeMeanSignal_ ) != string::npos ) { return sistrip::DAQ_SCOPE_MODE_MEAN_SIGNAL; }
  
  // unknown
  else if ( mon.find( sistrip::undefinedMonitorable_ ) != string::npos ) { return sistrip::UNDEFINED_MONITORABLE; }
  else { return sistrip::UNKNOWN_MONITORABLE; }
  
}  

// -----------------------------------------------------------------------------
// 
string SiStripHistoNamingScheme::presentation( const sistrip::Presentation& type ) {
  if ( type == sistrip::SUMMARY_HISTO ) { return sistrip::summaryHisto_; } 
  else if ( type == sistrip::SUMMARY_1D ) { return sistrip::summary1D_; }
  else if ( type == sistrip::SUMMARY_2D ) { return sistrip::summary2D_; }
  else if ( type == sistrip::SUMMARY_PROF )  { return sistrip::summaryProf_; }
  else if ( type == sistrip::UNDEFINED_PRESENTATION ) { return sistrip::undefinedPresentation_; }
  else { return sistrip::unknownPresentation_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::Presentation SiStripHistoNamingScheme::presentation( const string& type ) {
  if ( type.find( sistrip::summaryHisto_ ) != string::npos ) { return sistrip::SUMMARY_HISTO; } 
  else if ( type.find( sistrip::summary1D_ ) != string::npos ) { return sistrip::SUMMARY_1D; }
  else if ( type.find( sistrip::summary2D_ ) != string::npos ) { return sistrip::SUMMARY_2D; }
  else if ( type.find( sistrip::summaryProf_ ) != string::npos ) { return sistrip::SUMMARY_PROF; }
  else if ( type.find( sistrip::undefinedPresentation_ ) != string::npos ) { return sistrip::UNDEFINED_PRESENTATION; }
  else { return sistrip::UNKNOWN_PRESENTATION; }
}

// -----------------------------------------------------------------------------
// 
std::string SiStripHistoNamingScheme::cablingSource( const sistrip::CablingSource& source ) {
  if ( source == sistrip::CABLING_FROM_CONNS ) { return sistrip::cablingFromConns_; } 
  else if ( source == sistrip::CABLING_FROM_DEVICES ) { return sistrip::cablingFromDevices_; }
  else if ( source == sistrip::CABLING_FROM_DETIDS ) { return sistrip::cablingFromDetIds_; }
  else if ( source == sistrip::UNDEFINED_CABLING_SOURCE ) { return sistrip::undefinedCablingSource_; }
  else { return sistrip::unknownCablingSource_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::CablingSource SiStripHistoNamingScheme::cablingSource( const std::string& source ) {
  if ( source == "CONNECTIONS" || 
       source.find( sistrip::cablingFromConns_ ) != string::npos ) { return sistrip::CABLING_FROM_CONNS; }
  else if ( source == "DEVICES" || 
	    source.find( sistrip::cablingFromDevices_ ) != string::npos ) { return sistrip::CABLING_FROM_DEVICES; }
  else if ( source == "DETIDS" || 
	    source.find( sistrip::cablingFromDetIds_ ) != string::npos ) { return sistrip::CABLING_FROM_DETIDS; }
  else if ( source == "UNDEFINED" || 
	    source.find( sistrip::undefinedCablingSource_ ) != string::npos ) { return sistrip::UNDEFINED_CABLING_SOURCE; }
  else { return sistrip::UNKNOWN_CABLING_SOURCE; }
}



