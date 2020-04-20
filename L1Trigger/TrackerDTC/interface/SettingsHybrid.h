#ifndef L1Trigger_TrackerDTC_SettingsHybrid_h
#define L1Trigger_TrackerDTC_SettingsHybrid_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

namespace trackerDTC {

  class Settings;

  // Hybrid data format specific settings
  class SettingsHybrid {
    friend class Settings;

  public:
    enum SensorType { barrelPS, barrel2S, diskPS, disk2S, numSensorTypes };

    SettingsHybrid(const edm::ParameterSet& iConfig, Settings* settings);
    ~SettingsHybrid() {}

    void checkConfiguration(Settings* settings) const;

    void createEncodingsBend(Settings* settings);
    void createEncodingsLayer(Settings* settings);

    // TTStubalgo parameter

    double baseWindowSize() const { return baseWindowSize_; }

    // format specific parameter

    int widthR(SensorType type) const { return widthsR_[type]; }
    int widthPhi(SensorType type) const { return widthsPhi_[type]; }
    int widthZ(SensorType type) const { return widthsZ_[type]; }
    int widthAlpha(SensorType type) const { return widthsAlpha_[type]; }
    int widthBend(SensorType type) const { return widthsBend_[type]; }
    const std::vector<int>& numRingsPS() const { return numRingsPS_; }
    const std::vector<double>& layerRs() const { return layerRs_; }
    const std::vector<double>& diskZs() const { return diskZs_; }

    // derived format specific parameter

    double baseR(SensorType type) const { return basesR_[type]; }
    double basePhi(SensorType type) const { return basesPhi_[type]; }
    double baseZ(SensorType type) const { return basesZ_[type]; }
    double baseAlpha(SensorType type) const { return basesAlpha_[type]; }
    int numUnusedBits(SensorType type) const { return numsUnusedBits_[type]; }

    // derived TTStubalgo parameter

    const std::vector<double>& numTiltedLayerRings() const { return numTiltedLayerRings_; }
    const std::vector<double>& windowSizeBarrelLayers() const { return windowSizeBarrelLayers_; }
    const std::vector<std::vector<double>>& windowSizeTiltedLayerRings() const { return windowSizeTiltedLayerRings_; }
    const std::vector<std::vector<double>>& windowSizeEndcapDisksRings() const { return windowSizeEndcapDisksRings_; }

    // Hybrid specific encodings

    const std::vector<std::vector<int>>& layerIdEncodings() const { return layerIdEncodings_; }
    const std::vector<std::vector<double>>& bendEncodingsPS() const { return bendEncodingsPS_; }
    const std::vector<std::vector<double>>& bendEncodings2S() const { return bendEncodings2S_; }
    double disk2SR(int disk, int index) const { return disk2SRs_.at(disk).at(index); }

  private:
    //TrackerDTCFormat parameter sets

    const edm::ParameterSet paramsFormat_;
    const edm::ParameterSet paramsTTStubAlgo_;

    // TTStubAlgo parameter

    // check consitency between configured TTStub algo and the one used during input sample production
    const bool checkHistory_;
    // producer name used during input sample production
    const std::string productLabel_;
    // process name used during input sample production
    const std::string processName_;
    // precision of window sizes in pitch units
    const double baseWindowSize_;

    // format specific parameter

    // number of outer PS rings for disk 1-5
    const std::vector<int> numRingsPS_;
    // number of bits used for stub r w.r.t layer/disk centre
    const std::vector<int> widthsR_;
    // number of bits used for stub z w.r.t layer/disk centre
    const std::vector<int> widthsZ_;
    // number of bits used for stub phi w.r.t. region centre
    const std::vector<int> widthsPhi_;
    // number of bits used for stub row number
    const std::vector<int> widthsAlpha_;
    // number of bits used for stub bend number
    const std::vector<int> widthsBend_;
    // range in stub r which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    const std::vector<double> rangesR_;
    // range in stub z which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    const std::vector<double> rangesZ_;
    // range in stub row which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    const std::vector<double> rangesAlpha_;
    // mean radius of outer tracker barrel layer
    const std::vector<double> layerRs_;
    // mean z of outer tracker endcap disks
    const std::vector<double> diskZs_;
    // center radius of outer tracker endcap 2S diks strips
    const std::vector<edm::ParameterSet> disk2SRsSet_;

    // derived format specific parameter

    // precision or r in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> basesR_;
    // precision or phi in rad for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> basesPhi_;
    // precision or z in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> basesZ_;
    // precision or alpha in pitch units for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> basesAlpha_;
    // number of padded 0s in output data format for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> numsUnusedBits_;
    // center radius of outer tracker endcap 2S diks strips
    std::vector<std::vector<double>> disk2SRs_;

    // derived TTStubalgo parameter

    std::vector<double> numTiltedLayerRings_;
    std::vector<double> windowSizeBarrelLayers_;
    std::vector<std::vector<double>> windowSizeTiltedLayerRings_;
    std::vector<std::vector<double>> windowSizeEndcapDisksRings_;

    // Hybrid specific encodings

    // outer index = dtc id, inner index = encoded layer id, value = decoded layer id
    std::vector<std::vector<int>> layerIdEncodings_;
    // outer index = max window size in half strip units, inner index = decoded bend, value = encoded bend for PS modules
    std::vector<std::vector<double>> bendEncodingsPS_;
    // outer index = max window size in half strip units, inner index = decoded bend, value = encoded bend for 2S modules
    std::vector<std::vector<double>> bendEncodings2S_;
  };

}  // namespace trackerDTC

#endif