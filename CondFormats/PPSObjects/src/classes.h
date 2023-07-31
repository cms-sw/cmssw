#include "CondFormats/PPSObjects/src/headers.h"

namespace CondFormats_CTPPSPixelObjects {
  struct dictionary {
    std::map<CTPPSPixelFramePosition, CTPPSPixelROCInfo> ROCMapping;
    std::map<uint32_t, CTPPSPixelROCAnalysisMask> analysisMask;
    std::vector<CTPPSPixelGainCalibration::DetRegistry>::iterator p3;
    std::vector<CTPPSPixelGainCalibration::DetRegistry>::const_iterator p4;
    std::map<uint32_t, CTPPSPixelGainCalibration> mycalibmap;
    std::map<unsigned int, CTPPSRPAlignmentCorrectionData> mapType;

    //--- timing calibration parameters
    std::map<PPSTimingCalibration::Key, std::vector<double> > tc_tm;
    std::map<PPSTimingCalibration::Key, std::pair<double, double> > tc_pm;
    std::pair<PPSTimingCalibration::Key, std::vector<double> > tc_v_tm;
    std::pair<PPSTimingCalibration::Key, std::pair<double, double> > tc_v_pm;

    LHCOpticalFunctionsSet lhc_ofs;
    LHCOpticalFunctionsSetCollection lhc_ofsc;
    LHCInterpolatedOpticalFunctionsSet lhc_iofs;
    LHCInterpolatedOpticalFunctionsSetCollection lhc_iofsc;
    PPSPixelTopology pps_pt;

    
  };
}

// namespace {
//   struct dictionary {
//     std::map<TotemFramePosition, TotemVFATInfo> VFATMapping;
//     std::map<uint8_t, TotemDAQMapping::TotemTimingPlaneChannelPair> totemTimingChannelMap;
//     std::set<unsigned char> maskedChannels;
//     std::map<TotemSymbID, TotemVFATAnalysisMask> analysisMask;
//   }
// }

  // namespace CondFormats_CTPPSPixelObjects
