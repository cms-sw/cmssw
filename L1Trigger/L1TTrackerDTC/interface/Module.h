#ifndef __L1TTrackerDTC_MODULE__
#define __L1TTrackerDTC_MODULE__

#include "L1Trigger/L1TTrackerDTC/interface/Settings.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <vector>

namespace L1TTrackerDTC {

  // representation of an outer tracker sensormodule
  class Module {
    friend class Stub;

  public:
    Module(Settings* settings, const ::DetId& detId, const int& modId);

    ~Module() {}

    double r() const { return R_; }
    double phi() const { return Phi_; }
    double z() const { return Z_; }

  private:
    // handles 2 pi overflow
    double deltaPhi(const double& phi) { return reco::deltaPhi(phi, 0.); }

  private:
    int blockId_;    // outer tracker dtc routing block id [0-1]
    int channelId_;  // routing block channel id [0-35]

    bool side_;        // +z or -z
    bool barrel_;      // barrel or endcap
    bool flipped_;     // main sensor inside or outside
    bool signRow_;     // TTStub row needs flip of sign
    bool signCol_;     // TTStub col needs flip of sign
    bool signBend_;    // TTStub bend needs flip of sign
    int numColumns_;   // number of columns [2S=2,PS=8]
    int numRows_;      // number of rows [2S=8*127,PS=8*120]
    int layerId_;      // data format dependent
    double R_;         // module radius in cm
    double Phi_;       // module phi w.r.t. detector region centre in rad
    double Z_;         // module z in cm
    double sep_;       // sensor separation in cm
    double pitchRow_;  // sensor pitch in cm [strip=.009,pixel=.01]
    double pitchCol_;  // sensor length in cm [strip=5,pixel=.15625]
    double tilt_;      // module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double sin_;  // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double cos_;  // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis

    // hybrid format specific member

    SettingsHybrid::SensorType type_;   // module type (barrelPS, barrel2S, diskPS, disk2S)
    int decodedR_;                      // decoded radius of disk2S stubs
    double offsetR_;                    // stub radius offset for barrelPS, barrel2S
    double offsetZ_;                    // stub z offset for diskPS, disk2S
    std::vector<double> bendEncoding_;  // index = encoded bend, value = decoded bend
  };

}  // namespace L1TTrackerDTC

#endif
