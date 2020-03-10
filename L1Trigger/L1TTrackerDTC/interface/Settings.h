#ifndef __L1TTrackerDTC_SETTINGS_H__
#define __L1TTrackerDTC_SETTINGS_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include <vector>

namespace L1TTrackerDTC {

  class SettingsHybrid;
  class SettingsTMTT;

  // stores, calculates and provides run-time constants
  class Settings {
    friend class SettingsHybrid;
    friend class SettingsTMTT;

  public:
    Settings(const edm::ParameterSet& iConfig);

    ~Settings();

    // read in detector parameter
    void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);

    // ED parameter
    edm::InputTag inputTagTTStubDetSetVec() const { return inputTagTTStubDetSetVec_; }
    std::string productBranch() const { return productBranch_; }
    std::string dataFormat() const { return dataFormat_; }
    int offsetDetIdDSV() const { return offsetDetIdDSV_; }
    SettingsHybrid* hybrid() const { return hybrid_; }
    SettingsTMTT* tmtt() const { return tmtt_; }
    // Router parameter
    bool enableTruncation() const { return enableTruncation_; }
    int tmpTFP() const { return tmpTFP_; }
    int numFramesInfra() const { return numFramesInfra_; }
    int numRoutingBlocks() const { return numRoutingBlocks_; }
    int sizeStack() const { return sizeStack_; }
    // Converter parameter
    int widthRowLUT() const { return widthRowLUT_; }
    int widthQoverPt() const { return widthQoverPt_; }
    // Tracker parameter
    int numOverlappingRegions() const { return numOverlappingRegions_; }
    int numRegions() const { return numRegions_; }
    int numDTCsPerRegion() const { return numDTCsPerRegion_; }
    int numModulesPerDTC() const { return numModulesPerDTC_; }
    int widthBend() const { return widthBend_; }
    int widthCol() const { return widthCol_; }
    int widthRow() const { return widthRow_; }
    double baseBend() const { return baseBend_; }
    double baseCol() const { return baseCol_; }
    double baseRow() const { return baseRow_; }
    double bendCut() const { return bendCut_; }
    double freqLHC() const { return freqLHC_; }
    // format specific router parameter
    double maxEta() const { return maxEta_; }
    double minPt() const { return minPt_; }
    double chosenRofPhi() const { return chosenRofPhi_; }
    int numLayers() const { return numLayers_; }
    // f/w parameter
    double bField() const { return bField_; }
    // derived Router parameter
    int numDTCs() const { return numDTCs_; }
    int numModules() const { return numModules_; }
    int numModulesPerRoutingBlock() const { return numModulesPerRoutingBlock_; }
    int maxFramesChannelInput() const { return maxFramesChannelInput_; }
    int maxFramesChannelOutput() const { return maxFramesChannelOutput_; }
    // derived Converter parameter
    int widthR() const { return widthR_; }
    int widthPhi() const { return widthPhi_; }
    int widthZ() const { return widthZ_; }
    int numMergedRows() const { return numMergedRows_; }
    int widthLayer() const { return widthLayer_; }
    int widthM() const { return widthM_; }
    int widthC() const { return widthC_; }
    int widthEta() const { return widthEta_; }
    double rangeQoverPt() const { return rangeQoverPt_; }
    double maxCot() const { return maxCot_; }
    double maxQoverPt() const { return maxQoverPt_; }
    double baseRegion() const { return baseRegion_; }
    double baseQoverPt() const { return baseQoverPt_; }
    double baseR() const { return baseR_; }
    double baseZ() const { return baseZ_; }
    double basePhi() const { return basePhi_; }
    double baseM() const { return baseM_; }
    double baseC() const { return baseC_; }
    // event setup
    const std::vector< ::DetId> cablingMap() const { return cablingMap_; }
    const ::TrackerGeometry* trackerGeometry() const { return trackerGeometry_; }
    const ::TrackerTopology* trackerTopology() const { return trackerTopology_; }

  private:
    // DataFormats
    SettingsTMTT* tmtt_;
    SettingsHybrid* hybrid_;

    //TrackerDTCProducer parameter sets
    const edm::ParameterSet paramsED_;
    const edm::ParameterSet paramsRouter_;
    const edm::ParameterSet paramsConverter_;
    const edm::ParameterSet paramsTracker_;
    const edm::ParameterSet paramsFW_;
    const edm::ParameterSet paramsFormat_;

    // ED parameter
    const edm::InputTag inputTagTTStubDetSetVec_;
    const std::string ttStubAlgorithmProductLabel_;
    const std::string ttStubAlgorithmProcessName_;
    const std::string productBranch_;
    const std::string dataFormat_;  // "Hybrid" and "TMTT" format supported
    const int offsetDetIdDSV_;      // tk layout det id minus DetSetVec->detId
    const int offsetDetIdTP_;       // tk layout det id minus TrackerTopology lower det id

    // router parameter
    const double enableTruncation_;  // enables emulation of truncation
    const double freqDTC_;           // Frequency in MHz, has to be integer multiple of FreqLHC
    const int tmpTFP_;               // time multiplexed period of track finding processor
    const int numFramesInfra_;       // needed gap between events of emp-infrastructure firmware
    const int numRoutingBlocks_;     // number of systiloic arrays in stub router firmware
    const int sizeStack_;            // fifo depth in stub router firmware

    // converter parameter
    const int widthRowLUT_;   // number of row bits used in look up table
    const int widthQoverPt_;  // number of bits used for stub qOverPt

    // Tracker parameter
    const int numRegions_;             // number of phi slices the outer tracker readout is organized in
    const int numOverlappingRegions_;  // number of regions a reconstructable particles may cross
    const int numDTCsPerRegion_;       // number of DTC boards used to readout a detector region
    const int numModulesPerDTC_;       // max number of sensor modules connected to one DTC board
    const int tmpFE_;                  // number of events collected in front-end
    const int widthBend_;              // number of bits used for internal stub bend
    const int widthCol_;               // number of bits used for internal stub column
    const int widthRow_;               // number of bits used for internal stub row
    const double baseBend_;            // precision of internal stub bend in pitch units
    const double baseCol_;             // precision of internal stub column in pitch units
    const double baseRow_;             // precision of internal stub row in pitch units
    const double bendCut_;             // used stub bend uncertainty in pitch units
    const double freqLHC_;             // LHC bunch crossing rate in MHz

    // f/w constants
    const double speedOfLight_;  // in e8 m/s
    const double bField_;        // in T
    const double bFieldError_;   // accepted difference to EventSetup in T
    const double outerRadius_;   // outer radius of outer tracker in cm
    const double innerRadius_;   // inner radius of outer tracker in cm
    const double maxPitch_;      // max strip/pixel pitch of outer tracker sensors in cm

    // format specific parameter
    const double maxEta_;        // cut on stub eta
    const double minPt_;         // cut on stub pt, also defines region overlap shape in GeV
    const double chosenRofPhi_;  // critical radius defining region overlap shape in cm
    // max number of detector layer connected to one DTC (hybrid) number of detector layers a reconstructbale particle may cross (tmtt)
    const int numLayers_;

    // derived router parameter
    int numDTCs_;                    // total number of outer tracker DTCs
    int numModules_;                 // total number of max possible outer tracker modules (72 per DTC)
    int numModulesPerRoutingBlock_;  // number of inputs per systolic arrays in dtc firmware
    int maxFramesChannelInput_;      // max number of incomming stubs per packet (emp limit not cic limit)
    int maxFramesChannelOutput_;     // max number out outgoing stubs per packet

    // derived Converter parameter
    int widthR_;           // number of bits used for internal stub r - ChosenRofPhi
    int widthPhi_;         // number of bits used for internal stub phi w.r.t. region centre
    int widthZ_;           // number of bits used for internal stub z
    int numMergedRows_;    // number of merged rows for look up
    int widthLayer_;       // number of bits used for stub layer id
    int widthM_;           // number of bits used for phi of row slope
    int widthC_;           // number of bits used for phi or row intercept
    int widthEta_;         // number of bits used for internal stub eta
    double rangeQoverPt_;  // range of internal stub q over pt
    double maxCot_;        // cut on stub cot
    double maxQoverPt_;    // cut on stub q over pt
    double baseRegion_;    // region size in rad
    double baseQoverPt_;   // internal stub q over pt precision in 1 /cm
    double baseR_;         // internal stub r precision in cm
    double baseZ_;         // internal stub z precision in cm
    double basePhi_;       // internal stub phi precision in rad
    double baseM_;         // phi of row slope precision in rad / pitch unit
    double baseC_;         // phi of row intercept precision in rad

    // event setup
    std::vector< ::DetId> cablingMap_;  // index = track trigger module id [0-15551], value = det id
    const ::TrackerGeometry* trackerGeometry_;
    const ::TrackerTopology* trackerTopology_;
  };

  // Hybrid data format specific settings
  class SettingsHybrid {
    friend class Settings;

  public:
    SettingsHybrid(const edm::ParameterSet& iConfig, Settings* settings);

    ~SettingsHybrid() {}

    void beginRun(const edm::Run& iRun, Settings* settings);

    enum SensorType { barrelPS, barrel2S, diskPS, disk2S };

    // TTStubalgo parameter
    double baseWindowSize() const { return baseWindowSize_; }

    // format specific parameter
    int widthR(const SensorType& type) const { return widthsR_[type]; }
    int widthPhi(const SensorType& type) const { return widthsPhi_[type]; }
    int widthZ(const SensorType& type) const { return widthsZ_[type]; }
    int widthAlpha(const SensorType& type) const { return widthsAlpha_[type]; }
    int widthBend(const SensorType& type) const { return widthsBend_[type]; }
    std::vector<int> numRingsPS() const { return numRingsPS_; }
    std::vector<double> layerRs() const { return layerRs_; }
    std::vector<double> diskZs() const { return diskZs_; }

    // derived format specific parameter
    double baseR(const SensorType& type) const { return basesR_[type]; }
    double basePhi(const SensorType& type) const { return basesPhi_[type]; }
    double baseZ(const SensorType& type) const { return basesZ_[type]; }
    double baseAlpha(const SensorType& type) const { return basesAlpha_[type]; }
    int numUnusedBits(const SensorType& type) const { return numsUnusedBits_[type]; }

    // derived TTStubalgo parameter
    std::vector<double> numTiltedLayerRings() const { return numTiltedLayerRings_; }
    std::vector<double> windowSizeBarrelLayers() const { return windowSizeBarrelLayers_; }
    std::vector<std::vector<double> > windowSizeTiltedLayerRings() const { return windowSizeTiltedLayerRings_; }
    std::vector<std::vector<double> > windowSizeEndcapDisksRings() const { return windowSizeEndcapDisksRings_; }

    // Hybrid specific encodings
    std::vector<std::vector<int> > layerIdEncodings() const { return layerIdEncodings_; }
    std::vector<std::vector<double> > bendEncodingsPS() const { return bendEncodingsPS_; }
    std::vector<std::vector<double> > bendEncodings2S() const { return bendEncodings2S_; }

    double disk2SR(const int& disk, const int& index) const { return disk2SRs_.at(disk).at(index); }

  private:
    //TrackerDTCFormat parameter sets
    const edm::ParameterSet paramsFormat_;
    const edm::ParameterSet paramsTTStubAlgo_;

    // TTStubAlgo parameter
    const std::string productLabel_;  // TTStubAlgo producer name
    const std::string processName_;   // empty string possible if process unknown
    const double baseWindowSize_;     // precision of window sizes in pitch units

    // format specific parameter
    const std::vector<int> numRingsPS_;   // number of outer PS rings for disk 1-5
    const std::vector<int> widthsR_;      // number of bits used for stub r w.r.t layer/disk centre
    const std::vector<int> widthsZ_;      // number of bits used for stub z w.r.t layer/disk centre
    const std::vector<int> widthsPhi_;    // number of bits used for stub phi w.r.t. region centre
    const std::vector<int> widthsAlpha_;  // number of bits used for stub row number
    const std::vector<int> widthsBend_;   // number of bits used for stub bend number
    // range in stub r which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    const std::vector<double> rangesR_;
    // range in stub z which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    const std::vector<double> rangesZ_;
    // range in stub row which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    const std::vector<double> rangesAlpha_;
    const std::vector<double> layerRs_;                 // mean radius of outer tracker barrel layer
    const std::vector<double> diskZs_;                  // mean z of outer tracker endcap disks
    const std::vector<edm::ParameterSet> disk2SRsSet_;  // center radius of outer tracker endcap 2S diks strips

    // derived format specific parameter
    std::vector<double> basesR_;      // precision or r in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> basesPhi_;    // precision or phi in rad for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> basesZ_;      // precision or z in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> basesAlpha_;  // precision or alpha in pitch units for (barrelPS, barrel2S, diskPS, disk2S)
    // number of padded 0s in output data format for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> numsUnusedBits_;
    std::vector<std::vector<double> > disk2SRs_;  // center radius of outer tracker endcap 2S diks strips

    // derived TTStubalgo parameter
    std::vector<double> numTiltedLayerRings_;
    std::vector<double> windowSizeBarrelLayers_;
    std::vector<std::vector<double> > windowSizeTiltedLayerRings_;
    std::vector<std::vector<double> > windowSizeEndcapDisksRings_;

    // Hybrid specific encodings
    // outer index = dtc id, inner index = encoded layer id, value = decoded layer id
    std::vector<std::vector<int> > layerIdEncodings_;
    // outer index = max window size in half strip units, inner index = decoded bend, value = encoded bend for PS modules
    std::vector<std::vector<double> > bendEncodingsPS_;
    // outer index = max window size in half strip units, inner index = decoded bend, value = encoded bend for 2S modules
    std::vector<std::vector<double> > bendEncodings2S_;
  };

  // TMTT data format specific settings
  class SettingsTMTT {
    friend class Settings;

  public:
    SettingsTMTT(const edm::ParameterSet& iConfig, Settings* settings);

    ~SettingsTMTT() {}

    // format specific parameter
    int numSectorsPhi() const { return numSectorsPhi_; }
    int numBinsQoverPt() const { return numBinsQoverPt_; }
    int numBinsPhiT() const { return numBinsPhiT_; }
    double chosenRofZ() const { return chosenRofZ_; }
    double beamWindowZ() const { return beamWindowZ_; }
    double bounderiesEta(const int& eta) const { return bounderiesEta_[eta]; }
    // format specific parameter
    int numSectorsEta() const { return numSectorsEta_; }
    int widthQoverPtBin() const { return widthQoverPtBin_; }
    int numUnusedBits() const { return numUnusedBits_; }
    double maxZT() const { return maxZT_; }
    double baseSector() const { return baseSector_; }
    double baseQoverPt() const { return baseQoverPtBin_; }

  private:
    //TrackerDTCFormat parameter sets
    const edm::ParameterSet paramsFormat_;

    // format specific parameter
    const int numSectorsPhi_;                  // number of phi sectors used during track finding
    const int numBinsQoverPt_;                 // number of qOverPt bins used during track finding
    const int numBinsPhiT_;                    // number of phiT bins used during track finding
    const double chosenRofZ_;                  // critical radius defining r-z sector shape in cm
    const double beamWindowZ_;                 // half lumi region size in cm
    const double halfLength_;                  // has to be >= max stub z / 2 in cm
    const std::vector<double> bounderiesEta_;  // defining r-z sector shape

    // derived format specific parameter
    int numSectorsEta_;      // number of eta sectors used during track finding
    int widthQoverPtBin_;    // number of bits used for stub q over pt
    int numUnusedBits_;      // number of padded 0s in output data format
    double maxZT_;           // cut on zT
    double baseSector_;      // width of phi sector in rad
    double baseQoverPtBin_;  // precision of qOverPt bins used during track finding
  };

}  // namespace L1TTrackerDTC

#endif
