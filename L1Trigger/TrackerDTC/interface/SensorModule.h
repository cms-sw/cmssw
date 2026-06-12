#ifndef L1Trigger_TrackerDTC_SensorModule_h
#define L1Trigger_TrackerDTC_SensorModule_h

#include "DataFormats/DetId/interface/DetId.h"

#include <set>
#include <vector>
#include <cmath>

namespace trackerDTC {

  class Setup;

  /*! \class  trackerDTC::SensorModule
   *  \brief  Class to represent an outer tracker sensormodule
   *  \author Thomas Schuh
   *  \date   2025, Dec
   */
  class SensorModule {
  public:
    SensorModule(const Setup*,
                 const DetId&,
                 int,
                 int,
                 bool,
                 bool,
                 int,
                 int,
                 const std::vector<double>&,
                 const std::vector<double>&,
                 const std::set<int>&);
    ~SensorModule() {}

    enum Type { BarrelPS, Barrel2S, DiskPS, Disk2S, NumTypes };
    enum TypeTilt { nonBarrel = 0, tiltedMinus = 1, tiltedPlus = 2, flat = 3 };

    // cmssw det id
    const DetId& detId() const { return detId_; }
    // module type (BarrelPS, Barrel2S, DiskPS, Disk2S)
    Type type() const { return type_; }
    // dtc id [0-215]
    int dtcId() const { return dtcId_; }
    // module on dtc id [0-71]
    int modId() const { return modId_; }
    // +z or -z
    bool side() const { return side_; }
    // barrel or endcap
    bool barrel() const { return barrel_; }
    // tilted barrel or flat barrel
    bool tilted() const { return tilted_; }
    // Pixel-Strip or 2Strip module
    bool psModule() const { return psModule_; }
    // main sensor inside or outside
    bool flipped() const { return flipped_; }
    // TTStub row needs flip of sign
    bool signRow() const { return signRow_; }
    // TTStub col needs flip of sign
    bool signCol() const { return signCol_; }
    // TTStub bend needs flip of sign
    bool signBend() const { return signBend_; }
    // number of columns [2S=2,PS=8]
    int numColumns() const { return numColumns_; }
    // number of rows [2S=8*127,PS=8*120]
    int numRows() const { return numRows_; }
    // layer index [0-5,0-4]
    int layerIndex() const { return layerIndex_; }
    // layer index combined [barrel: 0-5, endcap: 6-10]
    int layerIndexCombined() const { return layerIndexCombined_; }
    // layer id [1-6,11-15]
    int layerId() const { return layerId_; }
    // layer id [0-6]
    int layerIdReduced() const { return layerIdReduced_; }
    // module radius in cm
    double r() const { return r_; }
    // module phi w.r.t. detector region centre in rad
    double phi() const { return phi_; }
    // module z in cm
    double z() const { return z_; }
    // sensor separation in cm
    double sep() const { return sep_; }
    // sensor pitch in cm [strip=.009,pixel=.01]
    double pitchRow() const { return pitchRow_; }
    // sensor length in cm [strip=5,pixel=.15625]
    double pitchCol() const { return pitchCol_; }
    // module tilt angle measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double tilt() const { return tilt_; }
    // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double sinTilt() const { return sinTilt_; }
    // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis
    double cosTilt() const { return cosTilt_; }
    // stub radius offset for barrelPS, barrel2S, used in Hybrid
    double offsetR() const { return offsetR_; }
    // stub z offset for diskPS, disk2S, used in Hybrid
    double offsetZ() const { return offsetZ_; }
    // bend window size in half pitch units
    int windowSize() const { return windowSize_; }
    // bend encoding
    int decodeBend(double) const;
    // bend encoding
    double encodeBend(int) const;
    // bend dergadadtion
    double degradeBend(int) const;
    // layer encoding
    int encodedLayer() const { return encodedLayer_; }
    // stub r uncertainty in cm
    double dR() const { return dR_; }
    // stub z uncertainty in cm for given track cot
    double dZ(double) const;
    // stub phi uncertainty in rad for given stub radius and track inv2R
    double dPhi(double, double) const;
    // tilt correction parameter used to project r to z uncertainty
    double tiltCorrectionSlope() const { return tiltCorrectionSlope_; }
    // tilt correction parameter used to project r to z uncertainty
    double tiltCorrectionIntercept() const { return tiltCorrectionIntercept_; }

  private:
    // module type
    int module() const;
    // cmssw det id
    DetId detId_;
    // dtc id [0-215]
    int dtcId_;
    // module on dtc id [0-71]
    int modId_;
    // +z or -z
    bool side_;
    // barrel or endcap
    bool barrel_;
    // tilted barrel or flat barrel
    bool tilted_;
    // Pixel-Strip or 2Strip module
    bool psModule_;
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
    // layer index [0-5,0-4]
    int layerIndex_;
    // layer index combined [barrel: 0-5, endcap: 6-10]
    int layerIndexCombined_;
    // layer id [1-6,11-15]
    int layerId_;
    // layer id [0-6]
    int layerIdReduced_;
    // module radius in cm
    double r_;
    // module phi w.r.t. detector region centre in rad
    double phi_;
    // module z in cm
    double z_;
    // sensor separation in cm
    double sep_;
    // sensor pitch in cm [strip=.009,pixel=.01]
    double pitchRow_;
    // sensor length in cm [strip=5,pixel=.15625]
    double pitchCol_;
    // module tilt angle measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double tilt_;
    // sinus of module tilt measured w.r.t. beam axis (0=barrel), tk layout measures w.r.t. radial axis
    double sinTilt_;
    // cosinus of module tilt measured w.r.t. beam axis (+-1=endcap), tk layout measures w.r.t. radial axis
    double cosTilt_;
    // module type (barrelPS, barrel2S, diskPS, disk2S)
    Type type_;
    // stub radius offset for barrelPS, barrel2S, used in Hybrid
    double offsetR_;
    // stub z offset for diskPS, disk2S, used in Hybrid
    double offsetZ_;
    // bend window size in half pitch units
    int windowSize_;
    // index = encoded bend, value = decoded bend for this module
    std::vector<double> encodingBend_;
    // index = decoded bend, value = degraded bend for this module
    std::vector<double> degradedBend_;
    // layer encoding
    int encodedLayer_;
    // tilt correction parameter used to project r to z uncertainty
    double tiltCorrectionSlope_;
    // tilt correction parameter used to project r to z uncertainty
    double tiltCorrectionIntercept_;
    // scattering term for dPhi
    double scattering_;
    // assumed average cluster width for dPhi
    double clusterWidth_;
    // additional phi uncertainties in rad
    double addPhiUncertainty_;
    // radial uncertainty
    double dR_;
  };

}  // namespace trackerDTC

#endif
