#ifndef DataFormats_SiStripCommon_SiStripConstants_H
#define DataFormats_SiStripCommon_SiStripConstants_H

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "boost/cstdint.hpp"
#include <string>

// -------------------- Some generic constants --------------------

namespace sistrip { 
  
  static const uint16_t invalid_ = 0xFFFF; // 65535
  static const uint16_t unknown_ = 0xFFFE; // 65534
  static const uint16_t maximum_ = 0x3FF;  // 1023
  static const std::string null_ = "null";
}

// -------------------- MessageLogger categories --------------------

namespace sistrip { 
  
  static const std::string mlDigis_       = "SiStripDigis";
  static const std::string mlCabling_     = "SiStripCabling";
  static const std::string mlConfigDb_    = "SiStripConfigDb";
  static const std::string mlRawToDigi_   = "SiStripRawToDigi";
  static const std::string mlDqmCommon_   = "SiStripDqmCommon";
  static const std::string mlDqmSource_   = "SiStripDqmSource";
  static const std::string mlDqmClient_   = "SiStripDqmClient";
  static const std::string mlInputSource_ = "SiStripInputSource";

}

// -------------------- FED- and control-related constants --------------------

namespace sistrip { 

  // FED id ranges
  static const uint16_t FED_ID_MIN  = static_cast<uint16_t>(FEDNumbering::getSiStripFEDIds().first);
  static const uint16_t FED_ID_MAX  = static_cast<uint16_t>(FEDNumbering::getSiStripFEDIds().second);
  static const uint16_t FED_ID_LAST = static_cast<uint16_t>(FEDNumbering::lastFEDId());

  // Front-End Driver, Front-End Unit, Fed Channels
  static const uint16_t FEDCH_PER_FEUNIT = 12;
  static const uint16_t FEUNITS_PER_FED  = 8;
  static const uint16_t FEDCH_PER_FED    = FEDCH_PER_FEUNIT * FEUNITS_PER_FED; // 96
  
  // APV25 front-end readout chip
  static const uint16_t APVS_PER_FEDCH   = 2;
  static const uint16_t APVS_PER_FEUNIT  = APVS_PER_FEDCH * FEDCH_PER_FEUNIT; // 24
  static const uint16_t APVS_PER_FED     = APVS_PER_FEUNIT * FEUNITS_PER_FED; // 194
  
  // Detector strips 
  static const uint16_t STRIPS_PER_APV    = 128;
  static const uint16_t STRIPS_PER_FEDCH  = STRIPS_PER_APV * APVS_PER_FEDCH;
  static const uint16_t STRIPS_PER_FEUNIT = STRIPS_PER_FEDCH * FEDCH_PER_FEUNIT; // 3072
  static const uint16_t STRIPS_PER_FED    = STRIPS_PER_FEUNIT * FEUNITS_PER_FED; // 24576

  // LLD = Linear Laser Driver
  static const uint16_t APVS_PER_CHAN = 2;
  static const uint16_t CHANS_PER_LLD = 3;

  // FED buffer formatting
  static const uint16_t DAQ_HDR_SIZE        = 8;
  static const uint16_t TRK_HDR_SIZE        = 8;
  static const uint16_t FE_HDR_SIZE         = 16;
  static const uint16_t APV_ERROR_HDR_SIZE  = 24;
  static const uint16_t FULL_DEBUG_HDR_SIZE = 8 * FE_HDR_SIZE;

  // VME crates
  static const uint16_t SLOTS_PER_CRATE =  20;
  static const uint16_t CRATE_SLOT_MIN  =   1;
  static const uint16_t CRATE_SLOT_MAX  =  21;
  static const uint16_t FEC_CRATE_MIN   =   1;
  static const uint16_t FEC_CRATE_MAX   =   4;
  static const uint16_t FED_CRATE_MIN   =   1;
  static const uint16_t FED_CRATE_MAX   =  60;

  // Control system
  static const uint16_t FEC_RING_MIN    =   1;
  static const uint16_t FEC_RING_MAX    =   8;
  static const uint16_t CCU_ADDR_MIN    =   1;
  static const uint16_t CCU_ADDR_MAX    = 127;
  static const uint16_t CCU_CHAN_MIN    =  16;
  static const uint16_t CCU_CHAN_MAX    =  31;
  static const uint16_t LLD_CHAN_MIN    =   1;
  static const uint16_t LLD_CHAN_MAX    =   3;
  static const uint16_t APV_I2C_MIN     =  32;
  static const uint16_t APV_I2C_MAX     =  37;

}

// -------------------- "Sources" of cabling object info --------------------

namespace sistrip { 
  
  static const std::string unknownCablingSource_ = "UnknownCablingSource";
  static const std::string undefinedCablingSource_ = "UndefinedCablingSource";
  static const std::string cablingFromConns_ = "CablingFromConnections";
  static const std::string cablingFromDevices_ = "CablingFromDevices";
  static const std::string cablingFromDetIds_ = "CablingFromDetIds";

}

// -------------------- DQM histograms --------------------

namespace sistrip { 
  
  // general
  static const std::string dqmRoot_ = "DQMData";
  static const std::string root_ = "SiStrip";
  static const std::string dir_ = "/";
  static const std::string sep_ = "_";
  static const std::string pipe_ = "|";
  static const std::string dot_ = ".";
  static const std::string hex_ = "0x";
  static const std::string commissioningTask_ = "SiStripCommissioningTask";
  
  // views
  static const std::string controlView_ = "ControlView";
  static const std::string readoutView_ = "ReadoutView";
  static const std::string detectorView_ = "DetectorView";
  static const std::string unknownView_ = "UnknownView";
  static const std::string undefinedView_ = "UndefinedView";

  // commissioning task
  static const std::string fedCabling_ = "FedCabling";
  static const std::string apvTiming_ = "ApvTiming";
  static const std::string fedTiming_ = "FedTiming";
  static const std::string optoScan_ = "OptoScan";
  static const std::string vpspScan_ = "VpspScan";
  static const std::string pedestals_ = "Pedestals";
  static const std::string apvLatency_ = "ApvLatency";
  static const std::string daqScopeMode_ = "DaqScopeMode";
  static const std::string physics_ = "Physics";
  static const std::string undefinedTask_ = "UndefinedTask";
  static const std::string unknownTask_ = "UnknownTask";

  // key 
  static const std::string fedKey_ = "FedKey";
  static const std::string fecKey_ = "FecKey";
  static const std::string detKey_ = "DetKey";
  //static const std::string noKey_ = "";
  static const std::string undefinedKey_ = "UndefinedKey";
  static const std::string unknownKey_ = "UnknownKey";

  // system granularity
  static const std::string tracker_ = "Tracker";
  static const std::string partition_ = "Partition";
  static const std::string tib_ = "Tib";
  static const std::string tob_ = "Tob";
  static const std::string tec_ = "Tec";

  // control granularity
  static const std::string fecCrate_ = "FecCrate";
  static const std::string fecSlot_ = "FecSlot";
  static const std::string fecRing_ = "FecRing";
  static const std::string ccuAddr_ = "CcuAddr";
  static const std::string ccuChan_ = "CcuChan";

  // readout granularity
  static const std::string fedId_ = "FedId";
  static const std::string feUnit_ = "FedFeUnit";
  static const std::string fedFeChan_ = "FedFeChan";
  static const std::string fedChannel_ = "FedChannel";

  // sub-structure granularity
  static const std::string layer_ = "Layer";
  static const std::string rod_ = "Rod";
  static const std::string string_ = "String";
  static const std::string disk_ = "Disk";
  static const std::string petal_ = "Petal";
  static const std::string ring_ = "Ring";

  // module granularity  
  static const std::string module_ = "Module";
  static const std::string lldChan_ = "LldChannel";
  static const std::string apv_ = "Apv";

  // misc granularity
  //static const std::string noGranularity_ = "";
  static const std::string undefinedGranularity_ = "UndefinedGranularity";
  static const std::string unknownGranularity_ = "UnknownGranularity";

  // extra histogram information 
  static const std::string gain_ = "Gain";
  static const std::string digital_ = "Digital";
  static const std::string pedsAndRawNoise_ = "PedsAndRawNoise";
  static const std::string residualsAndNoise_ = "ResidualsAndNoise";
  static const std::string commonMode_ = "CommonMode";

  // summary histogram types
  static const std::string summaryDistr_ = "SummaryDistr";
  static const std::string summary1D_ = "Summary1D";
  static const std::string summary2D_ = "Summary2D";
  static const std::string summaryProf_ = "SummaryProfile";
  static const std::string unknownSummaryType_ = "UnknownSummaryType";
  static const std::string undefinedSummaryType_ = "UndefinedSummaryType";

  // summary histogram names (general)
  static const std::string summaryHisto_ = "SummaryHisto";
  static const std::string unknownSummaryHisto_ = "UnknownSummaryHisto";
  static const std::string undefinedSummaryHisto_ = "UndefinedSummaryHisto";
  
  // summary histo names (fed cabling)
  static const std::string fedCablingFedId_ = "FedCablingFedId";
  static const std::string fedCablingFedCh_ = "FedCablingFedCh";
  static const std::string fedCablingSignalLevel_ = "FedCablingSignalLevel";

  // summary histo names (apv timing)
  static const std::string apvTimingTime_ = "TimingDelay";
  static const std::string apvTimingMax_ = "ApvTimingMax";
  static const std::string apvTimingDelay_ = "ApvTimingDelay";
  static const std::string apvTimingError_ = "ApvTimingError";
  static const std::string apvTimingBase_ = "ApvTimingBase";
  static const std::string apvTimingPeak_ = "ApvTimingPeak";
  static const std::string apvTimingHeight_ = "ApvTimingHeight";

  // summary histo names (fed timing)
  static const std::string fedTimingTime_ = "FedTimingTime";
  static const std::string fedTimingMax_ = "FedTimingMax";
  static const std::string fedTimingDelay_ = "FedTimingDelay";
  static const std::string fedTimingError_ = "FedTimingError";
  static const std::string fedTimingBase_ = "FedTimingBase";
  static const std::string fedTimingPeak_ = "FedTimingPeak";
  static const std::string fedTimingHeight_ = "FedTimingHeight";

  // summary histo names (opto scan)
  static const std::string optoScanLldBias_ = "OptoScanLldBias";
  static const std::string optoScanLldGain_ = "OptoScanLldGain";
  static const std::string optoScanMeasGain_ = "OptoScanMeasuredGain";
  static const std::string optoScanZeroLight_ = "OptoScanZeroLight";
  static const std::string optoScanLinkNoise_ = "OptoScanLinkNoise";
  static const std::string optoScanBaseLiftOff_ = "OptoScanBaseLiftOff";
  static const std::string optoScanLaserThresh_ = "OptoScanLaserThresh";
  static const std::string optoScanTickHeight_ = "OptoScanTickHeight";

  // summary histo names (vpsp scan)
  static const std::string vpspScanBothApvs_ = "VpspScanBothApvs";
  static const std::string vpspScanApv0_ = "VpspScanApv0";
  static const std::string vpspScanApv1_ = "VpspScanApv1";

  // summary histo names (pedestals)
  static const std::string pedestalsAllStrips_ = "Pedestals_AllStrips";
  static const std::string pedestalsMean_ = "Pedestals_Mean";
  static const std::string pedestalsSpread_ = "Pedestals_Spread";
  static const std::string pedestalsMax_ = "Pedestals_Max";
  static const std::string pedestalsMin_ = "Pedestals_Min";
  static const std::string noiseAllStrips_ = "Noise_AllStrips";
  static const std::string noiseMean_ = "Noise_Mean";
  static const std::string noiseSpread_ = "Noise_Spread";
  static const std::string noiseMax_ = "Noise_Max";
  static const std::string noiseMin_ = "Noise_Min";
  static const std::string numOfDead_ = "NumOfDead_Strips";
  static const std::string numOfNoisy_ = "NumOfNoisy_Strips";

  // summary histo names (daq scope mode)
  static const std::string daqScopeModeMeanSignal_ = "DaqScopeMode_MeanSignal";
  
}
  
#endif // DataFormats_SiStripCommon_SiStripConstants_H


