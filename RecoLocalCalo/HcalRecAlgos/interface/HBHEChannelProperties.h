#ifndef RecoLocalCalo_HcalRecAlgos_HBHEChannelProperties_h_
#define RecoLocalCalo_HcalRecAlgos_HBHEChannelProperties_h_

#include <array>
#include <cassert>

#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEPipelinePedestalAndGain.h"

class HcalCalibrations;
class HcalRecoParam;
class HcalQIECoder;
class HcalQIEShape;
class HcalSiPMParameter;

// Collection of HCAL HB/HE channel information, for faster lookup
// of this information in reco. This struct does not own any pointers.
struct HBHEChannelProperties {
  inline HBHEChannelProperties()
    : calib(nullptr), paramTs(nullptr), channelCoder(nullptr),
      shape(nullptr), siPMParameter(nullptr), pedestalsUpdated(false),
      taggedBadByDb(false), qualityUpdated(false) {}

  inline HBHEChannelProperties(
    const HcalCalibrations* i_calib,
    const HcalRecoParam* i_paramTs,
    const HcalQIECoder* i_channelCoder,
    const HcalQIEShape* i_shape,
    const HcalSiPMParameter* i_siPMParameter,
    const std::array<HBHEPipelinePedestalAndGain, 4>& i_pedsAndGains,
    const bool i_taggedBadByDb)
    : calib(i_calib), paramTs(i_paramTs), channelCoder(i_channelCoder),
      shape(i_shape), siPMParameter(i_siPMParameter),
      pedsAndGains(i_pedsAndGains), pedestalsUpdated(true),
      taggedBadByDb(i_taggedBadByDb), qualityUpdated(true) {
    assert(calib);
    assert(paramTs);
    assert(channelCoder);
    assert(shape);
    // siPMParameter is allowed to be nullptr (for QIE8)
  }

  const HcalCalibrations* calib;
  const HcalRecoParam* paramTs;
  const HcalQIECoder* channelCoder;
  const HcalQIEShape* shape;
  const HcalSiPMParameter* siPMParameter;
  std::array<HBHEPipelinePedestalAndGain, 4> pedsAndGains;
  bool pedestalsUpdated;

  bool taggedBadByDb;
  bool qualityUpdated;
};

#endif // RecoLocalCalo_HcalRecAlgos_HBHEChannelProperties_h_
