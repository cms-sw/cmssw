
#ifndef DataFormats_SiStripCommon_ConstantsForHardwareSystems_H
#define DataFormats_SiStripCommon_ConstantsForHardwareSystems_H

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/SiStripCommon/interface/Constants.h"

/**
   @file ConstantsForHardwareSystems.h
   @brief Constants and enumerated types for FED/FEC systems
*/

namespace sistrip {

  // -------------------- FED misc --------------------

  static const uint16_t FED_ADC_RANGE = 0x400;  // 1024

  // -------------------- FED ids --------------------

  static const uint16_t FED_ID_MIN = static_cast<uint16_t>(FEDNumbering::MINSiStripFEDID);
  static const uint16_t FED_ID_MAX = static_cast<uint16_t>(FEDNumbering::MAXSiStripFEDID);
  static const uint16_t CMS_FED_ID_MAX = static_cast<uint16_t>(FEDNumbering::MAXFEDID);
  static const uint16_t NUMBER_OF_FEDS = static_cast<uint16_t>(FED_ID_MAX - FED_ID_MIN + 1);

  // -------------------- FEDs to channels --------------------

  static const uint16_t FEDCH_PER_FEUNIT = 12;
  static const uint16_t FEUNITS_PER_FED = 8;
  static const uint16_t FEDCH_PER_FED = FEDCH_PER_FEUNIT * FEUNITS_PER_FED;  // 96

  // -------------------- Front-end devices --------------------

  static const uint16_t APVS_PER_FEDCH = 2;
  static const uint16_t APVS_PER_FEUNIT = APVS_PER_FEDCH * FEDCH_PER_FEUNIT;  // 24
  static const uint16_t APVS_PER_FED = APVS_PER_FEUNIT * FEUNITS_PER_FED;     // 194

  static const uint16_t APVS_PER_CHAN = 2;
  static const uint16_t CHANS_PER_LLD = 3;

  // -------------------- Detector strips --------------------

  static const uint16_t STRIPS_PER_APV = 128;
  static const uint16_t STRIPS_PER_FEDCH = STRIPS_PER_APV * APVS_PER_FEDCH;
  static const uint16_t STRIPS_PER_FEUNIT = STRIPS_PER_FEDCH * FEDCH_PER_FEUNIT;  // 3072
  static const uint16_t STRIPS_PER_FED = STRIPS_PER_FEUNIT * FEUNITS_PER_FED;     // 24576
  static constexpr float MeVperADCStrip = 9.5665E-4;

  // -------------------- FED buffers --------------------

  static const uint16_t DAQ_HDR_SIZE = 8;
  static const uint16_t TRK_HDR_SIZE = 8;
  static const uint16_t FE_HDR_SIZE = 16;
  static const uint16_t APV_ERROR_HDR_SIZE = 24;
  static const uint16_t FULL_DEBUG_HDR_SIZE = 8 * FE_HDR_SIZE;

  // -------------------- Control system --------------------

  //static const uint16_t FEC_CRATE_OFFSET =  1; //@@ temporary
  //static const uint16_t FEC_RING_OFFSET  =  1; //@@ temporary

  static const uint16_t FEC_RING_MIN = 1;
  static const uint16_t FEC_RING_MAX = 8;

  static const uint16_t CCU_ADDR_MIN = 1;
  static const uint16_t CCU_ADDR_MAX = 127;

  static const uint16_t CCU_CHAN_MIN = 16;
  static const uint16_t CCU_CHAN_MAX = 31;

  static const uint16_t LLD_CHAN_MIN = 1;
  static const uint16_t LLD_CHAN_MAX = 3;

  static const uint16_t APV_I2C_MIN = 32;
  static const uint16_t APV_I2C_MAX = 37;

  // -------------------- VME crates --------------------

  static const uint16_t SLOTS_PER_CRATE = 20;

  static const uint16_t CRATE_SLOT_MIN = 2;  // slot 1 is reserved for VME controller
  static const uint16_t CRATE_SLOT_MAX = 21;

  static const uint16_t MAX_FEDS_PER_CRATE = 16;
  static const uint16_t MAX_FECS_PER_CRATE = 20;

  static const uint16_t FED_CRATE_MIN = 1;
  static const uint16_t FED_CRATE_MAX = 60;

  static const uint16_t FEC_CRATE_MIN = 1;
  static const uint16_t FEC_CRATE_MAX = 4;

  // -------------------- String constants --------------------

  static const char unknownApvReadoutMode_[] = "UnknownApvReadoutMode";
  static const char undefinedApvReadoutMode_[] = "UndefinedApvReadoutMode";

  static const char apvPeakMode_[] = "ApvPeakMode";
  static const char apvDeconMode_[] = "ApvDeconMode";
  static const char apvMultiMode_[] = "ApvMultiMode";

  static const char unknownFedReadoutMode_[] = "UnknownFedReadoutMode";
  static const char undefinedFedReadoutMode_[] = "UndefinedFedReadoutMode";

  static const char fedScopeMode_[] = "FedScopeMode";
  static const char fedVirginRaw_[] = "FedVirginRaw";
  static const char fedProcRaw_[] = "FedProcessedRaw";
  static const char fedZeroSuppr_[] = "FedZeroSuppressed";
  static const char fedZeroSupprCMO_[] = "FedZeroSuppressedCMOverride";
  static const char fedZeroSupprLite_[] = "FedZeroSupprressedLite";
  static const char fedZeroSupprLiteCMO_[] = "FedZeroSuppressedLiteCMOverride";
  static const char fedZeroSupprLite8TT_[] = "FedZeroSuppressedLite8TT";
  static const char fedZeroSupprLite8TTCMO_[] = "FedZeroSuppressedLite8TTCMOverride";
  static const char fedZeroSupprLite8TB_[] = "FedZeroSuppressedLite8TB";
  static const char fedZeroSupprLite8TBCMO_[] = "FedZeroSuppressedLite8TBCMOverride";
  static const char fedZeroSupprLite8BB_[] = "FedZeroSuppressedLite8BB";
  static const char fedZeroSupprLite8BBCMO_[] = "FedZeroSuppressedLite8BBCMOverride";
  static const char fedPreMixRaw_[] = "FedPreMixRaw";

  // -------------------- Enumerators --------------------

  enum ApvReadoutMode {
    UNKNOWN_APV_READOUT_MODE = sistrip::unknown_,
    UNDEFINED_APV_READOUT_MODE = sistrip::invalid_,
    APV_PEAK_MODE = 1,
    APV_DECON_MODE = 2,
    APV_MULTI_MODE = 3
  };

  enum FedReadoutMode {
    UNKNOWN_FED_READOUT_MODE = sistrip::unknown_,
    UNDEFINED_FED_READOUT_MODE = sistrip::invalid_,
    FED_SCOPE_MODE = 1,
    FED_VIRGIN_RAW = 2,
    FED_PROC_RAW = 6,
    FED_ZERO_SUPPR = 10,
    FED_ZERO_SUPPR_LITE = 3,
    FED_ZERO_SUPPR_LITE_CMO = 4,
    FED_ZERO_SUPPR_LITE8_TT = 12,
    FED_ZERO_SUPPR_LITE8_TT_CMO = 8,
    FED_ZERO_SUPPR_LITE8_TB = 5,
    FED_ZERO_SUPPR_LITE8_TB_CMO = 7,
    FED_ZERO_SUPPR_LITE8_BB = 9,
    FED_ZERO_SUPPR_LITE8_BB_CMO = 11,
    FED_PREMIX_RAW = 15
  };

  enum FedReadoutPath {
    UNKNOWN_FED_READOUT_PATH = sistrip::unknown_,
    UNDEFINED_FED_READOUT_PATH = sistrip::invalid_,
    VME_READOUT = 1,
    SLINK_READOUT = 2
  };

  enum FedBufferFormat {
    UNKNOWN_FED_BUFFER_FORMAT = sistrip::unknown_,
    UNDEFINED_FED_BUFFER_FORMAT = sistrip::invalid_,
    FULL_DEBUG_FORMAT = 1,
    APV_ERROR_FORMAT = 2
  };

  enum FedSuperMode {
    UNKNOWN_FED_SUPER_MODE = sistrip::unknown_,
    UNDEFINED_FED_SUPER_MODE = sistrip::invalid_,
    REAL = 0,
    FAKE = 1
  };

}  // namespace sistrip

#endif  // DataFormats_SiStripCommon_ConstantsForHardwareSystems_H
