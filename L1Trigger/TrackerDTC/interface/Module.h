#ifndef L1Trigger_TrackerDTC_Module_h
#define L1Trigger_TrackerDTC_Module_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackerDTC/interface/SettingsHybrid.h"

#include <vector>

namespace trackerDTC {

  class Settings;

  // representation of an outer tracker sensormodule
  class Module {
    friend class Stub;

  public:
    Module(Settings* settings, const ::DetId& detId, int dtcId);
    ~Module() {}

  private:
    // handles 2 pi overflow
    double deltaPhi(double phi) { return reco::deltaPhi(phi, 0.); }

    // detector region [0-8]
    int region_;
    // +z or -z
    bool side_;
    // barrel or endcap
    bool barrel_;
    // main sensor inside or outside
    bool flipped_;
    // TTStub row needs flip of sign
    bool signRow_;
    // TTStub col needs flip of sign
    bool signCol_;
    // TTStub bend needs flip of sign
    bool signBend_;
    // number of columns [2S=2,PS=8]
    int numColumns_;
    // number of rows [2S=8*127,PS=8*120]
    int numRows_;
    // layer id [1-6,11-15]
    int layerId_;
    // module radius in cm
    double R_;
    // module phi w.r.t. detector region centre in rad
    double Phi_;
    // module z in cm
    double Z_;
    // sensor separation in cm
    double sep_;
    // sensor pitch in cm [strip=.009,pixel=.01]
    double pitchRow_;
    // sensor length in cm [strip=5,pixel=.15625]
    double pitchCol_;
    // module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double tilt_;
    // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double sin_;
    // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis
    double cos_;

    // hybrid format specific member

    // module type (barrelPS, barrel2S, diskPS, disk2S)
    SettingsHybrid::SensorType type_;
    // decoded radius of disk2S stubs
    int decodedR_;
    // stub radius offset for barrelPS, barrel2S
    double offsetR_;
    // stub z offset for diskPS, disk2S
    double offsetZ_;
    // index = encoded bend, value = decoded bend
    std::vector<double> bendEncoding_;
  };

}  // namespace trackerDTC

#endif
