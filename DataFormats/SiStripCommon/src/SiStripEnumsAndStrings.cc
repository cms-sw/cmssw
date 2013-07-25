// Last commit: $Id: SiStripEnumsAndStrings.cc,v 1.22 2010/02/22 14:10:23 wto Exp $

#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"

// -----------------------------------------------------------------------------
//
std::string SiStripEnumsAndStrings::view( const sistrip::View& view ) {
  if      ( view == sistrip::READOUT_VIEW ) { return sistrip::readoutView_; }
  else if ( view == sistrip::CONTROL_VIEW ) { return sistrip::controlView_; }
  else if ( view == sistrip::DETECTOR_VIEW ) { return sistrip::detectorView_; }
  else if ( view == sistrip::UNDEFINED_VIEW ) { return sistrip::undefinedView_; }
  else { return sistrip::unknownView_; }
}

// -----------------------------------------------------------------------------
//
sistrip::View SiStripEnumsAndStrings::view( const std::string& dir ) {
  if      ( dir.find( sistrip::readoutView_ ) != std::string::npos ) { return sistrip::READOUT_VIEW; } 
  else if ( dir.find( sistrip::controlView_ ) != std::string::npos ) { return sistrip::CONTROL_VIEW; } 
  else if ( dir.find( sistrip::detectorView_ ) != std::string::npos ) { return sistrip::DETECTOR_VIEW; } 
  else if ( dir.find( sistrip::undefinedView_ ) != std::string::npos ) { return sistrip::UNDEFINED_VIEW; } 
  else if ( dir.empty() ) { return sistrip::UNDEFINED_VIEW; } 
  else { return sistrip::UNKNOWN_VIEW; }
}

// -----------------------------------------------------------------------------
// 
std::string SiStripEnumsAndStrings::runType( const sistrip::RunType& run_type ) {
  if ( run_type == sistrip::FAST_CABLING ) { return sistrip::fastCablingRun_; }
  else if ( run_type == sistrip::FED_CABLING ) { return sistrip::fedCablingRun_; }
  else if ( run_type == sistrip::APV_TIMING ) { return sistrip::apvTimingRun_; }
  else if ( run_type == sistrip::FED_TIMING ) { return sistrip::fedTimingRun_; }
  else if ( run_type == sistrip::OPTO_SCAN ) { return sistrip::optoScanRun_; }
  else if ( run_type == sistrip::VPSP_SCAN ) { return sistrip::vpspScanRun_; }
  else if ( run_type == sistrip::PEDESTALS ) { return sistrip::pedestalsRun_; }
  else if ( run_type == sistrip::PEDS_ONLY ) { return sistrip::pedsOnlyRun_; }
  else if ( run_type == sistrip::NOISE ) { return sistrip::noiseRun_; }
  else if ( run_type == sistrip::PEDS_FULL_NOISE ) { return sistrip::pedsFullNoiseRun_; }
  else if ( run_type == sistrip::APV_LATENCY ){ return sistrip::apvLatencyRun_; }
  else if ( run_type == sistrip::FINE_DELAY ){ return sistrip::fineDelayRun_; }
  else if ( run_type == sistrip::CALIBRATION ){ return sistrip::calibPeakRun_; }
  else if ( run_type == sistrip::CALIBRATION_DECO ){ return sistrip::calibDeconRun_; }
  else if ( run_type == sistrip::CALIBRATION_SCAN ){ return sistrip::calibScanPeakRun_; }
  else if ( run_type == sistrip::CALIBRATION_SCAN_DECO ){ return sistrip::calibScanDeconRun_; }
  else if ( run_type == sistrip::DAQ_SCOPE_MODE ){ return sistrip::daqScopeModeRun_; }
  else if ( run_type == sistrip::PHYSICS ){ return sistrip::physicsRun_; }
  else if ( run_type == sistrip::UNDEFINED_RUN_TYPE ) { return sistrip::undefinedRunType_; }
  else { return sistrip::unknownRunType_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::RunType SiStripEnumsAndStrings::runType( const std::string& run_type ) {
  if ( run_type.find( sistrip::fastCablingRun_ ) != std::string::npos ) { return sistrip::FAST_CABLING; }
  else if ( run_type.find( sistrip::fedCablingRun_ ) != std::string::npos ) { return sistrip::FED_CABLING; }
  else if ( run_type.find( sistrip::apvTimingRun_ ) != std::string::npos ) { return sistrip::APV_TIMING; }
  else if ( run_type.find( sistrip::fedTimingRun_ ) != std::string::npos ) { return sistrip::FED_TIMING; }
  else if ( run_type.find( sistrip::optoScanRun_ ) != std::string::npos ) { return sistrip::OPTO_SCAN; }
  else if ( run_type.find( sistrip::vpspScanRun_ ) != std::string::npos ) { return sistrip::VPSP_SCAN; }
  else if ( run_type.find( sistrip::pedestalsRun_ ) != std::string::npos ) { return sistrip::PEDESTALS; }
  else if ( run_type.find( sistrip::pedsOnlyRun_ ) != std::string::npos ) { return sistrip::PEDS_ONLY; }
  else if ( run_type.find( sistrip::pedsFullNoiseRun_ ) != std::string::npos ) { return sistrip::PEDS_FULL_NOISE; } // needs to be before noise, otherwise name mismatch
  else if ( run_type.find( sistrip::noiseRun_ ) != std::string::npos ) { return sistrip::NOISE; }
  else if ( run_type.find( sistrip::fineDelayRun_ ) != std::string::npos ) { return sistrip::FINE_DELAY; }
  else if ( run_type.find( sistrip::calibPeakRun_ ) != std::string::npos ) { return sistrip::CALIBRATION; }
  else if ( run_type.find( sistrip::calibScanPeakRun_ ) != std::string::npos ) { return sistrip::CALIBRATION_SCAN; }
  else if ( run_type.find( sistrip::calibDeconRun_ ) != std::string::npos ) { return sistrip::CALIBRATION_DECO; }
  else if ( run_type.find( sistrip::calibScanDeconRun_ ) != std::string::npos ) { return sistrip::CALIBRATION_SCAN_DECO; }
  else if ( run_type.find( sistrip::apvLatencyRun_ ) != std::string::npos ) { return sistrip::APV_LATENCY; }
  else if ( run_type.find( sistrip::daqScopeModeRun_ ) != std::string::npos ) { return sistrip::DAQ_SCOPE_MODE; }
  else if ( run_type.find( sistrip::physicsRun_ ) != std::string::npos ) { return sistrip::PHYSICS; }
  else if ( run_type.find( sistrip::undefinedRunType_ ) != std::string::npos ) { return sistrip::UNDEFINED_RUN_TYPE; }
  else if ( run_type.empty() ) { return sistrip::UNDEFINED_RUN_TYPE; }
  else if ( run_type == "FAST_CABLING" ) { return sistrip::FAST_CABLING; }
  else if ( run_type == "FED_CABLING" ) { return sistrip::FED_CABLING; }
  else if ( run_type == "APV_TIMING" ) { return sistrip::APV_TIMING; }
  else if ( run_type == "FED_TIMING" ) { return sistrip::FED_TIMING; }
  else if ( run_type == "OPTO_SCAN" ) { return sistrip::OPTO_SCAN; }
  else if ( run_type == "VPSP_SCAN" ) { return sistrip::VPSP_SCAN; }
  else if ( run_type == "PEDESTALS" ) { return sistrip::PEDESTALS; }
  else if ( run_type == "PEDS_FULL_NOISE" ) { return sistrip::PEDS_FULL_NOISE; }
  else if ( run_type == "PEDS_ONLY" ) { return sistrip::PEDS_ONLY; }
  else if ( run_type == "NOISE" ) { return sistrip::NOISE; }
  else if ( run_type == "APV_LATENCY" ) { return sistrip::APV_LATENCY; }
  else if ( run_type == "FINE_DELAY" ) { return sistrip::FINE_DELAY; }
  else if ( run_type == "CALIBRATION" ) { return sistrip::CALIBRATION; }
  else if ( run_type == "CALIBRATION_SCAN" ) { return sistrip::CALIBRATION_SCAN; }
  else if ( run_type == "CALIBRATION_DECO" ) { return sistrip::CALIBRATION_DECO; }
  else if ( run_type == "CALIBRATION_SCAN_DECO" ) { return sistrip::CALIBRATION_SCAN_DECO; }
  else if ( run_type == "DAQ_SCOPE_MODE" ) { return sistrip::DAQ_SCOPE_MODE; }
  else if ( run_type == "PHYSICS" ) { return sistrip::PHYSICS; }
  else if ( run_type == "UNDEFINED" ) { return sistrip::UNDEFINED_RUN_TYPE; }
  else { return sistrip::UNKNOWN_RUN_TYPE; }
}

// -----------------------------------------------------------------------------
// 
sistrip::RunType SiStripEnumsAndStrings::runType( const uint16_t& run_type ) {
  if      ( run_type ==  1 ) { return sistrip::PHYSICS; }
  else if ( run_type ==  2 ) { return sistrip::PEDESTALS; }
  else if ( run_type ==  3 ) { return sistrip::CALIBRATION; }
  else if ( run_type ==  4 ) { return sistrip::OPTO_SCAN; }
  else if ( run_type ==  5 ) { return sistrip::APV_TIMING; }
  else if ( run_type ==  6 ) { return sistrip::APV_LATENCY; }
  else if ( run_type ==  7 ) { return sistrip::FINE_DELAY_PLL; }
  else if ( run_type ==  8 ) { return sistrip::FINE_DELAY_TTC; }
  else if ( run_type == 10 ) { return sistrip::MULTI_MODE; }
  else if ( run_type == 11 ) { return sistrip::FED_CABLING; }
  else if ( run_type == 12 ) { return sistrip::FED_TIMING; }
  else if ( run_type == 13 ) { return sistrip::FED_CABLING; }
  else if ( run_type == 14 ) { return sistrip::VPSP_SCAN; }
  else if ( run_type == 15 ) { return sistrip::DAQ_SCOPE_MODE; }
  else if ( run_type == 16 ) { return sistrip::QUITE_FAST_CABLING; }
  else if ( run_type == 17 ) { return sistrip::FINE_DELAY; } //@@ layer
  else if ( run_type == 18 ) { return sistrip::PHYSICS_ZS; }
  else if ( run_type == 19 ) { return sistrip::CALIBRATION_SCAN; }
  else if ( run_type == 20 ) { return sistrip::CALIBRATION_SCAN_DECO; }
  else if ( run_type == 21 ) { return sistrip::FAST_CABLING; }
  else if ( run_type == 22 ) { return sistrip::PEDS_ONLY; }
  else if ( run_type == 23 ) { return sistrip::NOISE; }
  else if ( run_type == 24 ) { return sistrip::PEDS_FULL_NOISE; }
  else if ( run_type == 33 ) { return sistrip::CALIBRATION_DECO; }
  else if ( run_type == sistrip::invalid_ ) { return sistrip::UNDEFINED_RUN_TYPE; }
  else { return sistrip::UNKNOWN_RUN_TYPE; }
}

// -----------------------------------------------------------------------------
// 
std::string SiStripEnumsAndStrings::keyType( const sistrip::KeyType& key_type ) {
  if ( key_type == sistrip::FED_KEY ) { return sistrip::fedKey_; }
  else if ( key_type == sistrip::FEC_KEY ) { return sistrip::fecKey_; }
  else if ( key_type == sistrip::DET_KEY ) { return sistrip::detKey_; }
  else if ( key_type == sistrip::UNDEFINED_KEY )  { return sistrip::undefinedKey_; }
  else { return sistrip::unknownKey_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::KeyType SiStripEnumsAndStrings::keyType( const std::string& key_type ) {
  if ( key_type.find ( sistrip::fedKey_) != std::string::npos ) { return sistrip::FED_KEY; }
  else if ( key_type.find( sistrip::fecKey_) != std::string::npos ) { return sistrip::FEC_KEY; }
  else if ( key_type.find( sistrip::detKey_) != std::string::npos ) { return sistrip::DET_KEY; }
  else if ( key_type.find( sistrip::undefinedKey_) != std::string::npos ) { return sistrip::UNDEFINED_KEY; }
  else if ( key_type.empty() ) { return sistrip::UNDEFINED_KEY; }
  else { return sistrip::UNKNOWN_KEY; }
}  

// -----------------------------------------------------------------------------
//
std::string SiStripEnumsAndStrings::granularity( const sistrip::Granularity& granularity ) {
  // System
  if ( granularity == sistrip::TRACKER ) { return sistrip::tracker_; }
  else if ( granularity == sistrip::PARTITION ) { return sistrip::partition_; }
  else if ( granularity == sistrip::TIB ) { return sistrip::tib_; }
  else if ( granularity == sistrip::TOB ) { return sistrip::tob_; }
  else if ( granularity == sistrip::TEC ) { return sistrip::tec_; }
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
  // Readout
  else if ( granularity == sistrip::FED_SYSTEM ) { return sistrip::fedSystem_; }
  else if ( granularity == sistrip::FE_DRIVER ) { return sistrip::feDriver_; }
  else if ( granularity == sistrip::FE_UNIT ) { return sistrip::feUnit_; }
  else if ( granularity == sistrip::FE_CHAN ) { return sistrip::feChan_; }
  else if ( granularity == sistrip::FED_APV ) { return sistrip::fedApv_; }
  else if ( granularity == sistrip::FED_CHANNEL ) { return sistrip::fedChannel_; }
  // Control
  else if ( granularity == sistrip::FEC_SYSTEM ) { return sistrip::fecSystem_; }
  else if ( granularity == sistrip::FEC_CRATE ) { return sistrip::fecCrate_; }
  else if ( granularity == sistrip::FEC_SLOT ) { return sistrip::fecSlot_; }
  else if ( granularity == sistrip::FEC_RING ) { return sistrip::fecRing_; }
  else if ( granularity == sistrip::CCU_ADDR ) { return sistrip::ccuAddr_; }
  else if ( granularity == sistrip::CCU_CHAN ) { return sistrip::ccuChan_; }
  // Unknown
  else if ( granularity == sistrip::UNDEFINED_GRAN ) { return sistrip::undefinedGranularity_; }
  else { return sistrip::unknownGranularity_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::Granularity SiStripEnumsAndStrings::granularity( const std::string& granularity ) {
  // System
  if ( granularity.find( sistrip::tracker_ ) != std::string::npos ) { return sistrip::TRACKER; }
  else if ( granularity.find( sistrip::partition_ ) != std::string::npos ) { return sistrip::PARTITION; }
  else if ( granularity.find( sistrip::tib_ ) != std::string::npos ) { return sistrip::TIB; }
  else if ( granularity.find( sistrip::tob_ ) != std::string::npos ) { return sistrip::TOB; }
  else if ( granularity.find( sistrip::tec_ ) != std::string::npos ) { return sistrip::TEC; }
  // Readout
  else if ( granularity.find( sistrip::fedSystem_ ) != std::string::npos ) { return sistrip::FED_SYSTEM; }
  else if ( granularity.find( sistrip::feDriver_ ) != std::string::npos ) { return sistrip::FE_DRIVER; }
  else if ( granularity.find( sistrip::feUnit_ ) != std::string::npos ) { return sistrip::FE_UNIT; }
  else if ( granularity.find( sistrip::feChan_ ) != std::string::npos ) { return sistrip::FE_CHAN; }
  else if ( granularity.find( sistrip::fedApv_ ) != std::string::npos ) { return sistrip::FED_APV; }
  else if ( granularity.find( sistrip::fedChannel_ ) != std::string::npos ) { return sistrip::FED_CHANNEL; }
  // Control
  else if ( granularity.find( sistrip::fecSystem_ ) != std::string::npos ) { return sistrip::FEC_SYSTEM; }
  else if ( granularity.find( sistrip::fecCrate_ ) != std::string::npos ) { return sistrip::FEC_CRATE; }
  else if ( granularity.find( sistrip::fecSlot_ ) != std::string::npos ) { return sistrip::FEC_SLOT; }
  else if ( granularity.find( sistrip::fecRing_ ) != std::string::npos ) { return sistrip::FEC_RING; }
  else if ( granularity.find( sistrip::ccuAddr_ ) != std::string::npos ) { return sistrip::CCU_ADDR; }
  else if ( granularity.find( sistrip::ccuChan_ ) != std::string::npos ) { return sistrip::CCU_CHAN; }
  // Sub-structure
  else if ( granularity.find( sistrip::layer_ ) != std::string::npos ) { return sistrip::LAYER; }
  else if ( granularity.find( sistrip::rod_ ) != std::string::npos ) { return sistrip::ROD; }
  else if ( granularity.find( sistrip::string_ ) != std::string::npos ) { return sistrip::STRING; }
  else if ( granularity.find( sistrip::disk_ ) != std::string::npos ) { return sistrip::DISK; }
  else if ( granularity.find( sistrip::petal_ ) != std::string::npos ) { return sistrip::PETAL; }
  else if ( granularity.find( sistrip::ring_ ) != std::string::npos ) { return sistrip::RING; }
  // Module and below
  else if ( granularity.find( sistrip::module_ ) != std::string::npos ) { return sistrip::MODULE; }
  else if ( granularity.find( sistrip::lldChan_ ) != std::string::npos ) { return sistrip::LLD_CHAN; }
  else if ( granularity.find( sistrip::apv_ ) != std::string::npos ) { return sistrip::APV; } //@@ bug if before "FedApv"!
  // Unknown
  else if ( granularity.find( sistrip::undefinedGranularity_ ) != std::string::npos ) { return sistrip::UNDEFINED_GRAN; }
  else if ( granularity.empty() ) { return sistrip::UNDEFINED_GRAN; }
  else { return sistrip::UNKNOWN_GRAN; }
}  

// -----------------------------------------------------------------------------
//
std::string SiStripEnumsAndStrings::apvReadoutMode( const sistrip::ApvReadoutMode& mode ) {
  if      ( mode == sistrip::APV_PEAK_MODE ) { return sistrip::apvPeakMode_; }
  else if ( mode == sistrip::APV_DECON_MODE ) { return sistrip::apvDeconMode_; }
  else if ( mode == sistrip::APV_MULTI_MODE ) { return sistrip::apvMultiMode_; }
  else if ( mode == sistrip::UNDEFINED_APV_READOUT_MODE ) { return sistrip::undefinedApvReadoutMode_; }
  else { return sistrip::unknownApvReadoutMode_; }
}

// -----------------------------------------------------------------------------
//
sistrip::ApvReadoutMode SiStripEnumsAndStrings::apvReadoutMode( const std::string& mode ) {
  if      ( mode.find( sistrip::apvPeakMode_ ) != std::string::npos ) { return sistrip::APV_PEAK_MODE; } 
  else if ( mode.find( sistrip::apvDeconMode_ ) != std::string::npos ) { return sistrip::APV_DECON_MODE; } 
  else if ( mode.find( sistrip::apvMultiMode_ ) != std::string::npos ) { return sistrip::APV_MULTI_MODE; } 
  else if ( mode.find( sistrip::undefinedApvReadoutMode_ ) != std::string::npos ) { return sistrip::UNDEFINED_APV_READOUT_MODE; } 
  else if ( mode.empty() ) { return sistrip::UNDEFINED_APV_READOUT_MODE; } 
  else { return sistrip::UNKNOWN_APV_READOUT_MODE; }
}

// -----------------------------------------------------------------------------
//
std::string SiStripEnumsAndStrings::fedReadoutMode( const sistrip::FedReadoutMode& mode ) {
  if      ( mode == sistrip::FED_SCOPE_MODE ) { return sistrip::fedScopeMode_; }
  else if ( mode == sistrip::FED_VIRGIN_RAW ) { return sistrip::fedVirginRaw_; }
  else if ( mode == sistrip::FED_PROC_RAW ) { return sistrip::fedProcRaw_; }
  else if ( mode == sistrip::FED_ZERO_SUPPR ) { return sistrip::fedZeroSuppr_; }
  else if ( mode == sistrip::FED_ZERO_SUPPR_LITE ) { return sistrip::fedZeroSupprLite_; }
  else if ( mode == sistrip::UNDEFINED_FED_READOUT_MODE ) { return sistrip::undefinedFedReadoutMode_; }
  else { return sistrip::unknownFedReadoutMode_; }
}

// -----------------------------------------------------------------------------
//
sistrip::FedReadoutMode SiStripEnumsAndStrings::fedReadoutMode( const std::string& mode ) {
  if      ( mode.find( sistrip::fedScopeMode_ ) != std::string::npos ) { return sistrip::FED_SCOPE_MODE; } 
  else if ( mode.find( sistrip::fedVirginRaw_ ) != std::string::npos ) { return sistrip::FED_VIRGIN_RAW; } 
  else if ( mode.find( sistrip::fedProcRaw_ ) != std::string::npos ) { return sistrip::FED_PROC_RAW; } 
  else if ( mode.find( sistrip::fedZeroSuppr_ ) != std::string::npos ) { return sistrip::FED_ZERO_SUPPR; } 
  else if ( mode.find( sistrip::fedZeroSupprLite_ ) != std::string::npos ) { return sistrip::FED_ZERO_SUPPR_LITE; } 
  else if ( mode.find( sistrip::undefinedFedReadoutMode_ ) != std::string::npos ) { return sistrip::UNDEFINED_FED_READOUT_MODE; } 
  else if ( mode.empty() ) { return sistrip::UNDEFINED_FED_READOUT_MODE; } 
  else { return sistrip::UNKNOWN_FED_READOUT_MODE; }
}

// -----------------------------------------------------------------------------
// 
std::string SiStripEnumsAndStrings::histoType( const sistrip::HistoType& type ) {
  if ( type == sistrip::EXPERT_HISTO ) { return sistrip::expertHisto_; } 
  else if ( type == sistrip::SUMMARY_HISTO ) { return sistrip::summaryHisto_; } 
  else if ( type == sistrip::UNDEFINED_HISTO_TYPE ) { return sistrip::undefinedHistoType_; }
  else { return sistrip::unknownHistoType_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::HistoType SiStripEnumsAndStrings::histoType( const std::string& type ) {
  if ( type.find( sistrip::expertHisto_ ) != std::string::npos ) { return sistrip::EXPERT_HISTO; }
  else if ( type.find( sistrip::summaryHisto_ ) != std::string::npos ) { return sistrip::SUMMARY_HISTO; }
  else if ( type.find( sistrip::undefinedHistoType_ ) != std::string::npos ) { return sistrip::UNDEFINED_HISTO_TYPE; } 
  else if ( type.empty() ) { return sistrip::UNDEFINED_HISTO_TYPE; } 
  else { return sistrip::UNKNOWN_HISTO_TYPE; }
}

// -----------------------------------------------------------------------------
// 
std::string SiStripEnumsAndStrings::monitorable( const sistrip::Monitorable& mon ) {
  
  // fed cabling
  if ( mon == sistrip::FED_CABLING_FED_ID ) { return sistrip::fedCablingFedId_; } 
  else if ( mon == sistrip::FED_CABLING_FED_CH ) { return sistrip::fedCablingFedCh_; } 
  else if ( mon == sistrip::FED_CABLING_ADC_LEVEL ) { return sistrip::fedCablingAdcLevel_; }

  // fast fed cabling
  else if ( mon == sistrip::FAST_CABLING_DCU_ID ) { return sistrip::fastCablingDcuId_; } 
  else if ( mon == sistrip::FAST_CABLING_LLD_CH ) { return sistrip::fastCablingLldCh_; } 
  else if ( mon == sistrip::FAST_CABLING_HIGH_LEVEL ) { return sistrip::fastCablingHighLevel_; }
  else if ( mon == sistrip::FAST_CABLING_HIGH_RMS ) { return sistrip::fastCablingHighRms_; }
  else if ( mon == sistrip::FAST_CABLING_LOW_LEVEL ) { return sistrip::fastCablingLowLevel_; }
  else if ( mon == sistrip::FAST_CABLING_LOW_RMS ) { return sistrip::fastCablingLowRms_; }
  else if ( mon == sistrip::FAST_CABLING_MAX ) { return sistrip::fastCablingMax_; }
  else if ( mon == sistrip::FAST_CABLING_MIN ) { return sistrip::fastCablingMin_; }
  else if ( mon == sistrip::FAST_CABLING_CONNS_PER_FED ) { return sistrip::fastCablingConnsPerFed_; }
  
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
  else if ( mon == sistrip::VPSP_SCAN_APV_SETTINGS ) { return sistrip::vpspScanBothApvs_; }
  else if ( mon == sistrip::VPSP_SCAN_APV0_SETTING ) { return sistrip::vpspScanApv0_; }
  else if ( mon == sistrip::VPSP_SCAN_APV1_SETTING ) { return sistrip::vpspScanApv1_; }
  else if ( mon == sistrip::VPSP_SCAN_ADC_LEVEL ) { return sistrip::vpspScanAdcLevel_; }
  else if ( mon == sistrip::VPSP_SCAN_DIGITAL_HIGH ) { return sistrip::vpspScanDigitalHigh_; }
  else if ( mon == sistrip::VPSP_SCAN_DIGITAL_LOW ) { return sistrip::vpspScanDigitalLow_; }

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
  else if ( mon == sistrip::NOISE_GAUS_ALL_STRIPS ) { return sistrip::noiseGausAllStrips_; }
  else if ( mon == sistrip::NOISE_KS_ALL_STRIPS ) { return sistrip::noiseKSAllStrips_; }
  else if ( mon == sistrip::NOISE_CHI2_ALL_STRIPS ) { return sistrip::noiseChi2AllStrips_; }
  else if ( mon == sistrip::NOISE_BIN_84_ALL_STRIPS) { return sistrip::noiseBin84AllStrips_; }
  else if ( mon == sistrip::NOISE_RMS_ALL_STRIPS) { return sistrip::noiseRMSAllStrips_; }
  else if ( mon == sistrip::NOISE_SIGNIF_ALL_STRIPS ) { return sistrip::noiseSignif_; }

  // fine delay
  else if ( mon == sistrip::FINE_DELAY_POS) { return sistrip::fineDelayPos_; }
  else if ( mon == sistrip::FINE_DELAY_ERROR) { return sistrip::fineDelayErr_; }

  // calibration
  else if ( mon == sistrip::CALIBRATION_AMPLITUDE) { return sistrip::calibrationAmplitude_; }
  else if ( mon == sistrip::CALIBRATION_TAIL) { return sistrip::calibrationTail_; }
  else if ( mon == sistrip::CALIBRATION_RISETIME) { return sistrip::calibrationRiseTime_; }
  else if ( mon == sistrip::CALIBRATION_TIMECONSTANT) { return sistrip::calibrationTimeConstant_; }
  else if ( mon == sistrip::CALIBRATION_SMEARING) { return sistrip::calibrationSmearing_; }
  else if ( mon == sistrip::CALIBRATION_CHI2) { return sistrip::calibrationChi2_; }
  else if ( mon == sistrip::CALIBRATION_AMPLITUDE_ALLSTRIPS) { return sistrip::calibrationAmplitudeAS_; }
  else if ( mon == sistrip::CALIBRATION_TAIL_ALLSTRIPS) { return sistrip::calibrationTailAS_; }
  else if ( mon == sistrip::CALIBRATION_RISETIME_ALLSTRIPS) { return sistrip::calibrationRiseTimeAS_; }
  else if ( mon == sistrip::CALIBRATION_TIMECONSTANT_ALLSTRIPS) { return sistrip::calibrationTimeConstantAS_; }
  else if ( mon == sistrip::CALIBRATION_SMEARING_ALLSTRIPS) { return sistrip::calibrationSmearingAS_; }
  else if ( mon == sistrip::CALIBRATION_CHI2_ALLSTRIPS) { return sistrip::calibrationChi2AS_; }
  else if ( mon == sistrip::CALIBRATION_AMPLITUDE_MIN) { return sistrip::calibrationAmplitudeMin_; }
  else if ( mon == sistrip::CALIBRATION_TAIL_MIN) { return sistrip::calibrationTailMin_; }
  else if ( mon == sistrip::CALIBRATION_RISETIME_MIN) { return sistrip::calibrationRiseTimeMin_; }
  else if ( mon == sistrip::CALIBRATION_TIMECONSTANT_MIN) { return sistrip::calibrationTimeConstantMin_; }
  else if ( mon == sistrip::CALIBRATION_SMEARING_MIN) { return sistrip::calibrationSmearingMin_; }
  else if ( mon == sistrip::CALIBRATION_CHI2_MIN) { return sistrip::calibrationChi2Min_; }
  else if ( mon == sistrip::CALIBRATION_AMPLITUDE_MAX) { return sistrip::calibrationAmplitudeMax_; }
  else if ( mon == sistrip::CALIBRATION_TAIL_MAX) { return sistrip::calibrationTailMax_; }
  else if ( mon == sistrip::CALIBRATION_RISETIME_MAX) { return sistrip::calibrationRiseTimeMax_; }
  else if ( mon == sistrip::CALIBRATION_TIMECONSTANT_MAX) { return sistrip::calibrationTimeConstantMax_; }
  else if ( mon == sistrip::CALIBRATION_SMEARING_MAX) { return sistrip::calibrationSmearingMax_; }
  else if ( mon == sistrip::CALIBRATION_CHI2_MAX) { return sistrip::calibrationChi2Max_; }
  
  // scope mode 
  else if ( mon == sistrip::DAQ_SCOPE_MODE_MEAN_SIGNAL ) { return sistrip::daqScopeModeMeanSignal_; }

  // unknown
  else if ( mon == sistrip::UNDEFINED_MONITORABLE ) { return sistrip::undefinedMonitorable_; }
  else { return sistrip::unknownMonitorable_; }
  
}

// -----------------------------------------------------------------------------
// 
sistrip::Monitorable SiStripEnumsAndStrings::monitorable( const std::string& mon ) {

  // fed cabling
  if ( mon.find( sistrip::fedCablingFedId_ ) != std::string::npos ) { return sistrip::FED_CABLING_FED_ID; } 
  else if ( mon.find( sistrip::fedCablingFedCh_ ) != std::string::npos ) { return sistrip::FED_CABLING_FED_CH; } 
  else if ( mon.find( sistrip::fedCablingAdcLevel_ ) != std::string::npos ) { return sistrip::FED_CABLING_ADC_LEVEL; } 

  // fed cabling
  else if ( mon.find( sistrip::fastCablingDcuId_ ) != std::string::npos ) { return sistrip::FAST_CABLING_DCU_ID; } 
  else if ( mon.find( sistrip::fastCablingLldCh_ ) != std::string::npos ) { return sistrip::FAST_CABLING_LLD_CH; } 
  else if ( mon.find( sistrip::fastCablingHighLevel_ ) != std::string::npos ) { return sistrip::FAST_CABLING_HIGH_LEVEL; } 
  else if ( mon.find( sistrip::fastCablingHighRms_ ) != std::string::npos ) { return sistrip::FAST_CABLING_HIGH_RMS; } 
  else if ( mon.find( sistrip::fastCablingLowLevel_ ) != std::string::npos ) { return sistrip::FAST_CABLING_LOW_LEVEL; } 
  else if ( mon.find( sistrip::fastCablingLowRms_ ) != std::string::npos ) { return sistrip::FAST_CABLING_LOW_RMS; } 
  else if ( mon.find( sistrip::fastCablingMax_ ) != std::string::npos ) { return sistrip::FAST_CABLING_MAX; } 
  else if ( mon.find( sistrip::fastCablingMin_ ) != std::string::npos ) { return sistrip::FAST_CABLING_MIN; } 
  else if ( mon.find( sistrip::fastCablingConnsPerFed_ ) != std::string::npos ) { return sistrip::FAST_CABLING_CONNS_PER_FED; }

  // apv timing
  else if ( mon.find( sistrip::apvTimingTime_ ) != std::string::npos ) { return sistrip::APV_TIMING_TIME; } 
  else if ( mon.find( sistrip::apvTimingMax_ ) != std::string::npos ) { return sistrip::APV_TIMING_MAX_TIME; }
  else if ( mon.find( sistrip::apvTimingDelay_ ) != std::string::npos ) { return sistrip::APV_TIMING_DELAY; }
  else if ( mon.find( sistrip::apvTimingError_ ) != std::string::npos ) { return sistrip::APV_TIMING_ERROR; }
  else if ( mon.find( sistrip::apvTimingBase_ ) != std::string::npos ) { return sistrip::APV_TIMING_BASE; }
  else if ( mon.find( sistrip::apvTimingPeak_ ) != std::string::npos ) { return sistrip::APV_TIMING_PEAK; }
  else if ( mon.find( sistrip::apvTimingHeight_ ) != std::string::npos ) { return sistrip::APV_TIMING_HEIGHT; }

  // fed timing
  else if ( mon.find( sistrip::fedTimingTime_ ) != std::string::npos ) { return sistrip::FED_TIMING_TIME; } 
  else if ( mon.find( sistrip::fedTimingMax_ ) != std::string::npos ) { return sistrip::FED_TIMING_MAX_TIME; }
  else if ( mon.find( sistrip::fedTimingDelay_ ) != std::string::npos ) { return sistrip::FED_TIMING_DELAY; }
  else if ( mon.find( sistrip::fedTimingError_ ) != std::string::npos ) { return sistrip::FED_TIMING_ERROR; }
  else if ( mon.find( sistrip::fedTimingBase_ ) != std::string::npos ) { return sistrip::FED_TIMING_BASE; }
  else if ( mon.find( sistrip::fedTimingPeak_ ) != std::string::npos ) { return sistrip::FED_TIMING_PEAK; }
  else if ( mon.find( sistrip::fedTimingHeight_ ) != std::string::npos ) { return sistrip::FED_TIMING_HEIGHT; }

  // opto scan
  else if ( mon.find( sistrip::optoScanLldGain_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_LLD_GAIN_SETTING; }
  else if ( mon.find( sistrip::optoScanLldBias_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_LLD_BIAS_SETTING; }
  else if ( mon.find( sistrip::optoScanMeasGain_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_MEASURED_GAIN; }
  else if ( mon.find( sistrip::optoScanZeroLight_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_ZERO_LIGHT_LEVEL; }
  else if ( mon.find( sistrip::optoScanLinkNoise_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_LINK_NOISE; }
  else if ( mon.find( sistrip::optoScanBaseLiftOff_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_BASELINE_LIFT_OFF; }
  else if ( mon.find( sistrip::optoScanLaserThresh_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_LASER_THRESHOLD; }
  else if ( mon.find( sistrip::optoScanTickHeight_ ) != std::string::npos ) { return sistrip::OPTO_SCAN_TICK_HEIGHT; }

  // vpsp scan
  else if ( mon.find( sistrip::vpspScanBothApvs_ ) != std::string::npos ) { return sistrip::VPSP_SCAN_APV_SETTINGS; }
  else if ( mon.find( sistrip::vpspScanApv0_ ) != std::string::npos ) { return sistrip::VPSP_SCAN_APV0_SETTING; }
  else if ( mon.find( sistrip::vpspScanApv1_ ) != std::string::npos ) { return sistrip::VPSP_SCAN_APV1_SETTING; }
  else if ( mon.find( sistrip::vpspScanAdcLevel_ ) != std::string::npos ) { return sistrip::VPSP_SCAN_ADC_LEVEL; }
  else if ( mon.find( sistrip::vpspScanDigitalHigh_ ) != std::string::npos ) { return sistrip::VPSP_SCAN_DIGITAL_HIGH; }
  else if ( mon.find( sistrip::vpspScanDigitalLow_ ) != std::string::npos ) { return sistrip::VPSP_SCAN_DIGITAL_LOW; }
  
  // pedestals / noise
  else if ( mon.find( sistrip::pedestalsAllStrips_ ) != std::string::npos ) { return sistrip::PEDESTALS_ALL_STRIPS; }
  else if ( mon.find( sistrip::pedestalsMean_ ) != std::string::npos ) { return sistrip::PEDESTALS_MEAN; }
  else if ( mon.find( sistrip::pedestalsSpread_ ) != std::string::npos ) { return sistrip::PEDESTALS_SPREAD; }
  else if ( mon.find( sistrip::pedestalsMax_ ) != std::string::npos ) { return sistrip::PEDESTALS_MAX; }
  else if ( mon.find( sistrip::pedestalsMin_ ) != std::string::npos ) { return sistrip::PEDESTALS_MIN; }

  // noiseKS must come before the more general noiseAllStrips, since it contains that string
  else if ( mon.find( sistrip::noiseGausAllStrips_ ) != std::string::npos ) { return sistrip::NOISE_GAUS_ALL_STRIPS; }
  else if ( mon.find( sistrip::noiseKSAllStrips_ ) != std::string::npos ) { return sistrip::NOISE_KS_ALL_STRIPS; }
  else if ( mon.find( sistrip::noiseChi2AllStrips_ ) != std::string::npos ) { return sistrip::NOISE_CHI2_ALL_STRIPS; }
  else if ( mon.find( sistrip::noiseBin84AllStrips_ ) != std::string::npos ) { return sistrip::NOISE_BIN_84_ALL_STRIPS;}
  else if ( mon.find( sistrip::noiseRMSAllStrips_ ) != std::string::npos ) { return sistrip::NOISE_RMS_ALL_STRIPS;}
  else if ( mon.find( sistrip::noiseSignif_ ) != std::string::npos ) { return sistrip::NOISE_SIGNIF_ALL_STRIPS; }
  else if ( mon.find( sistrip::noiseAllStrips_ ) != std::string::npos ) { return sistrip::NOISE_ALL_STRIPS; }
  else if ( mon.find( sistrip::noiseMean_ ) != std::string::npos ) { return sistrip::NOISE_MEAN; }
  else if ( mon.find( sistrip::noiseSpread_ ) != std::string::npos ) { return sistrip::NOISE_SPREAD; }
  else if ( mon.find( sistrip::noiseMax_ ) != std::string::npos ) { return sistrip::NOISE_MAX; }
  else if ( mon.find( sistrip::noiseMin_ ) != std::string::npos ) { return sistrip::NOISE_MIN; }
  else if ( mon.find( sistrip::numOfDead_ ) != std::string::npos ) { return sistrip::NUM_OF_DEAD; }
  else if ( mon.find( sistrip::numOfNoisy_ ) != std::string::npos ) { return sistrip::NUM_OF_NOISY; }

  // fine delay
  else if ( mon.find( sistrip::fineDelayPos_ ) != std::string::npos ) { return sistrip::FINE_DELAY_POS; }
  else if ( mon.find( sistrip::fineDelayErr_ ) != std::string::npos ) { return sistrip::FINE_DELAY_ERROR; }
 
  // calibration
  else if ( mon.find( sistrip::calibrationAmplitudeAS_ ) != std::string::npos ) { return sistrip::CALIBRATION_AMPLITUDE_ALLSTRIPS; }
  else if ( mon.find( sistrip::calibrationTailAS_ ) != std::string::npos ) { return sistrip::CALIBRATION_TAIL_ALLSTRIPS; }
  else if ( mon.find( sistrip::calibrationRiseTimeAS_ ) != std::string::npos ) { return sistrip::CALIBRATION_RISETIME_ALLSTRIPS; }
  else if ( mon.find( sistrip::calibrationTimeConstantAS_ ) != std::string::npos ) { return sistrip::CALIBRATION_TIMECONSTANT_ALLSTRIPS; }
  else if ( mon.find( sistrip::calibrationSmearingAS_ ) != std::string::npos ) { return sistrip::CALIBRATION_SMEARING_ALLSTRIPS; }
  else if ( mon.find( sistrip::calibrationChi2AS_ ) != std::string::npos ) { return sistrip::CALIBRATION_CHI2_ALLSTRIPS; }
  else if ( mon.find( sistrip::calibrationAmplitudeMin_ ) != std::string::npos ) { return sistrip::CALIBRATION_AMPLITUDE_MIN; }
  else if ( mon.find( sistrip::calibrationTailMin_ ) != std::string::npos ) { return sistrip::CALIBRATION_TAIL_MIN; }
  else if ( mon.find( sistrip::calibrationRiseTimeMin_ ) != std::string::npos ) { return sistrip::CALIBRATION_RISETIME_MIN; }
  else if ( mon.find( sistrip::calibrationTimeConstantMin_ ) != std::string::npos ) { return sistrip::CALIBRATION_TIMECONSTANT_MIN; }
  else if ( mon.find( sistrip::calibrationSmearingMin_ ) != std::string::npos ) { return sistrip::CALIBRATION_SMEARING_MIN; }
  else if ( mon.find( sistrip::calibrationChi2Min_ ) != std::string::npos ) { return sistrip::CALIBRATION_CHI2_MIN; }
  else if ( mon.find( sistrip::calibrationAmplitudeMax_ ) != std::string::npos ) { return sistrip::CALIBRATION_AMPLITUDE_MAX; }
  else if ( mon.find( sistrip::calibrationTailMax_ ) != std::string::npos ) { return sistrip::CALIBRATION_TAIL_MAX; }
  else if ( mon.find( sistrip::calibrationRiseTimeMax_ ) != std::string::npos ) { return sistrip::CALIBRATION_RISETIME_MAX; }
  else if ( mon.find( sistrip::calibrationTimeConstantMax_ ) != std::string::npos ) { return sistrip::CALIBRATION_TIMECONSTANT_MAX; }
  else if ( mon.find( sistrip::calibrationSmearingMax_ ) != std::string::npos ) { return sistrip::CALIBRATION_SMEARING_MAX; }
  else if ( mon.find( sistrip::calibrationChi2Max_ ) != std::string::npos ) { return sistrip::CALIBRATION_CHI2_MAX; }
  else if ( mon.find( sistrip::calibrationAmplitude_ ) != std::string::npos ) { return sistrip::CALIBRATION_AMPLITUDE; }
  else if ( mon.find( sistrip::calibrationTail_ ) != std::string::npos ) { return sistrip::CALIBRATION_TAIL; }
  else if ( mon.find( sistrip::calibrationRiseTime_ ) != std::string::npos ) { return sistrip::CALIBRATION_RISETIME; }
  else if ( mon.find( sistrip::calibrationTimeConstant_ ) != std::string::npos ) { return sistrip::CALIBRATION_TIMECONSTANT; }
  else if ( mon.find( sistrip::calibrationSmearing_ ) != std::string::npos ) { return sistrip::CALIBRATION_SMEARING; }
  else if ( mon.find( sistrip::calibrationChi2_ ) != std::string::npos ) { return sistrip::CALIBRATION_CHI2; }

  // scope mode
  else if ( mon.find( sistrip::daqScopeModeMeanSignal_ ) != std::string::npos ) { return sistrip::DAQ_SCOPE_MODE_MEAN_SIGNAL; }
  
  // unknown
  else if ( mon.find( sistrip::undefinedMonitorable_ ) != std::string::npos ) { return sistrip::UNDEFINED_MONITORABLE; }
  else if ( mon.empty() ) { return sistrip::UNDEFINED_MONITORABLE; }
  else { return sistrip::UNKNOWN_MONITORABLE; }
  
}  

// -----------------------------------------------------------------------------
// 
std::string SiStripEnumsAndStrings::presentation( const sistrip::Presentation& type ) {
  if ( type == sistrip::HISTO_1D ) { return sistrip::histo1d_; } 
  else if ( type == sistrip::HISTO_2D_SUM ) { return sistrip::histo2dSum_; }
  else if ( type == sistrip::HISTO_2D_SCATTER ) { return sistrip::histo2dScatter_; }
  else if ( type == sistrip::PROFILE_1D )  { return sistrip::profile1D_; }
  else if ( type == sistrip::UNDEFINED_PRESENTATION ) { return sistrip::undefinedPresentation_; }
  else { return sistrip::unknownPresentation_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::Presentation SiStripEnumsAndStrings::presentation( const std::string& type ) {
  if ( type.find( sistrip::histo1d_ ) != std::string::npos ) { return sistrip::HISTO_1D; } 
  else if ( type.find( sistrip::histo2dSum_ ) != std::string::npos ) { return sistrip::HISTO_2D_SUM; }
  else if ( type.find( sistrip::histo2dScatter_ ) != std::string::npos ) { return sistrip::HISTO_2D_SCATTER; }
  else if ( type.find( sistrip::profile1D_ ) != std::string::npos ) { return sistrip::PROFILE_1D; }
  else if ( type.find( sistrip::undefinedPresentation_ ) != std::string::npos ) { return sistrip::UNDEFINED_PRESENTATION; }
  else if ( type.empty() ) { return sistrip::UNDEFINED_PRESENTATION; }
  else { return sistrip::UNKNOWN_PRESENTATION; }
}

// -----------------------------------------------------------------------------
// 
std::string SiStripEnumsAndStrings::cablingSource( const sistrip::CablingSource& source ) {
  if ( source == sistrip::CABLING_FROM_CONNS ) { return sistrip::cablingFromConns_; } 
  else if ( source == sistrip::CABLING_FROM_DEVICES ) { return sistrip::cablingFromDevices_; }
  else if ( source == sistrip::CABLING_FROM_DETIDS ) { return sistrip::cablingFromDetIds_; }
  else if ( source == sistrip::UNDEFINED_CABLING_SOURCE ) { return sistrip::undefinedCablingSource_; }
  else { return sistrip::unknownCablingSource_; }
}

// -----------------------------------------------------------------------------
// 
sistrip::CablingSource SiStripEnumsAndStrings::cablingSource( const std::string& source ) {
  if ( source.find( sistrip::cablingFromConns_ ) != std::string::npos ) { return sistrip::CABLING_FROM_CONNS; }
  else if ( source.find( sistrip::cablingFromDevices_ ) != std::string::npos ) { return sistrip::CABLING_FROM_DEVICES; }
  else if ( source.find( sistrip::cablingFromDetIds_ ) != std::string::npos ) { return sistrip::CABLING_FROM_DETIDS; }
  else if ( source.find( sistrip::undefinedCablingSource_ ) != std::string::npos ) { return sistrip::UNDEFINED_CABLING_SOURCE; }
  else if ( source.empty() ) { return sistrip::UNDEFINED_CABLING_SOURCE; }
  else if ( source == "CONNECTIONS" ) { return sistrip::CABLING_FROM_CONNS; }
  else if ( source == "DEVICES" ) { return sistrip::CABLING_FROM_DEVICES; }
  else if ( source == "DETIDS" ) { return sistrip::CABLING_FROM_DETIDS; }
  else if ( source == "UNDEFINED" ) { return sistrip::UNDEFINED_CABLING_SOURCE; }
  else { return sistrip::UNKNOWN_CABLING_SOURCE; }
}

