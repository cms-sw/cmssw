#ifndef RecoLocalCalo_HcalRecAlgos_HcalChannelProperties_h_
#define RecoLocalCalo_HcalRecAlgos_HcalChannelProperties_h_

#include <array>
#include <vector>
#include <cassert>

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPipelinePedestalAndGain.h"

class HcalCalibrations;
class HcalRecoParam;
class HcalQIECoder;
class HcalQIEShape;
class HcalSiPMParameter;

// Collection of HCAL channel information, for faster lookup of this
// information in reco. This struct does not own any pointers.
struct HcalChannelProperties {
  inline HcalChannelProperties()
      : calib(nullptr),
        paramTs(nullptr),
        channelCoder(nullptr),
        shape(nullptr),
        siPMParameter(nullptr),
        taggedBadByDb(false) {}

  inline HcalChannelProperties(const HcalCalibrations* i_calib,
                               const HcalRecoParam* i_paramTs,
                               const HcalQIECoder* i_channelCoder,
                               const HcalQIEShape* i_shape,
                               const HcalSiPMParameter* i_siPMParameter,
                               const std::array<HcalPipelinePedestalAndGain, 4>& i_pedsAndGains,
                               const bool i_taggedBadByDb)
      : calib(i_calib),
        paramTs(i_paramTs),
        channelCoder(i_channelCoder),
        shape(i_shape),
        siPMParameter(i_siPMParameter),
        pedsAndGains(i_pedsAndGains),
        taggedBadByDb(i_taggedBadByDb) {
    assert(calib);
    assert(paramTs);
    assert(channelCoder);
    assert(shape);
    /* siPMParameter is allowed to be nullptr for QIE8 */
  }

  const HcalCalibrations* calib;
  const HcalRecoParam* paramTs;
  const HcalQIECoder* channelCoder;
  const HcalQIEShape* shape;
  const HcalSiPMParameter* siPMParameter;
  std::array<HcalPipelinePedestalAndGain, 4> pedsAndGains;
  bool taggedBadByDb;
};

typedef std::vector<HcalChannelProperties> HcalChannelPropertiesVec;

#endif  // RecoLocalCalo_HcalRecAlgos_HcalChannelProperties_h_
