#ifndef L1Trigger_TrackerDTC_Settings_h
#define L1Trigger_TrackerDTC_Settings_h

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"

#include "L1Trigger/TrackerDTC/interface/SettingsHybrid.h"
#include "L1Trigger/TrackerDTC/interface/SettingsTMTT.h"
#include "L1Trigger/TrackerDTC/interface/Module.h"

#include <vector>
#include <memory>
#include <unordered_map>

namespace trackerDTC {

  // stores, calculates and provides run-time constants
  class Settings {
    friend class SettingsHybrid;
    friend class SettingsTMTT;

  private:
    enum SubDetId { pixelBarrel = 1, pixelDisks = 2 };

  public:
    Settings(const edm::ParameterSet& iConfig);
    ~Settings() {}

    // store TrackerGeometry
    void setTrackerGeometry(const TrackerGeometry* trackerGeometry);
    // store TrackerTopology
    void setTrackerTopology(const TrackerTopology* trackerTopology);
    // store MagneticField
    void setMagneticField(const MagneticField* magneticField);
    // store TrackerDetToDTCELinkCablingMap
    void setCablingMap(const TrackerDetToDTCELinkCablingMap* cablingMap);
    // store TTStubAlgorithm handle
    void setTTStubAlgorithm(const edm::ESHandle<TTStubAlgorithm<Ref_Phase2TrackerDigi_>>& handleTTStubAlgorithm);
    // store GeometryConfiguration handle
    void setGeometryConfiguration(const edm::ESHandle<DDCompactView>& handleGeometryConfiguration);
    // store ProcessHistory
    void setProcessHistory(const edm::ProcessHistory& processHistory);
    // check current coniguration consistency with input configuration
    void checkConfiguration();
    // convert ES Products into handy objects
    void beginRun();
    // convert DetId to module id [0:15551]
    int modId(const ::DetId& detId) const;
    // collection of modules connected to a specific dtc
    const std::vector<Module*>& modules(int dtcId) const;

    // ED parameter

    const edm::InputTag& inputTagTTStubDetSetVec() const { return inputTagTTStubDetSetVec_; }
    const edm::ESInputTag& inputTagMagneticField() const { return inputTagMagneticField_; }
    const edm::ESInputTag& inputTagTrackerGeometry() const { return inputTagTrackerGeometry_; }
    const edm::ESInputTag& inputTagTrackerTopology() const { return inputTagTrackerTopology_; }
    const edm::ESInputTag& inputTagCablingMap() const { return inputTagCablingMap_; }
    const edm::ESInputTag& inputTagTTStubAlgorithm() const { return inputTagTTStubAlgorithm_; }
    const edm::ESInputTag& inputTagGeometryConfiguration() const { return inputTagGeometryConfiguration_; }
    const std::string& productBranchAccepted() const { return productBranchAccepted_; }
    const std::string& productBranchLost() const { return productBranchLost_; }
    const std::string& dataFormat() const { return dataFormat_; }
    // tk layout det id minus DetSetVec->detId
    int offsetDetIdDSV() const { return offsetDetIdDSV_; }
    // tk layout det id minus TrackerTopology lower det id
    int offsetDetIdTP() const { return offsetDetIdTP_; }
    // hybrid format specific configurations
    SettingsHybrid* hybrid() const { return hybrid_.get(); }
    // tmtt format specific configurations
    SettingsTMTT* tmtt() const { return tmtt_.get(); }
    // offset in layer ids between barrel layer and endcap disks
    int offsetLayerDisks() const { return offsetLayerDisks_; }
    // offset between 0 and smallest layer id (barrel layer 1)
    int offsetLayerId() const { return offsetLayerId_; }

    // Router parameter

    // enables emulation of truncation
    bool enableTruncation() const { return enableTruncation_; }
    // time multiplexed period of track finding processor
    int tmpTFP() const { return tmpTFP_; }
    // needed gap between events of emp-infrastructure firmware
    int numFramesInfra() const { return numFramesInfra_; }
    // number of systiloic arrays in stub router firmware
    int numRoutingBlocks() const { return numRoutingBlocks_; }
    // fifo depth in stub router firmware
    int sizeStack() const { return sizeStack_; }

    // Converter parameter

    // number of row bits used in look up table
    int widthRowLUT() const { return widthRowLUT_; }
    // number of bits used for stub qOverPt. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    int widthQoverPt() const { return widthQoverPt_; }

    // Tracker parameter

    // number of regions a reconstructable particles may cross
    int numOverlappingRegions() const { return numOverlappingRegions_; }
    // number of phi slices the outer tracker readout is organized in
    int numRegions() const { return numRegions_; }
    // number of DTC boards used to readout a detector region
    int numDTCsPerRegion() const { return numDTCsPerRegion_; }
    // max number of sensor modules connected to one DTC board
    int numModulesPerDTC() const { return numModulesPerDTC_; }
    // number of bits used for internal stub bend
    int widthBend() const { return widthBend_; }
    // number of bits used for internal stub column
    int widthCol() const { return widthCol_; }
    // number of bits used for internal stub row
    int widthRow() const { return widthRow_; }
    // precision of internal stub bend in pitch units
    double baseBend() const { return baseBend_; }
    // precision of internal stub column in pitch units
    double baseCol() const { return baseCol_; }
    // precision of internal stub row in pitch units
    double baseRow() const { return baseRow_; }
    // used stub bend uncertainty in pitch units
    double bendCut() const { return bendCut_; }
    // LHC bunch crossing rate in MHz
    double freqLHC() const { return freqLHC_; }

    // format specific router parameter

    // cut on stub eta
    double maxEta() const { return maxEta_; }
    // cut on stub pt, also defines region overlap shape in GeV
    double minPt() const { return minPt_; }
    // critical radius defining region overlap shape in cm
    double chosenRofPhi() const { return chosenRofPhi_; }
    // tmtt: number of detector layers a reconstructbale particle may cross
    // hybrid:  max number of detector layer connected to one DTC
    int numLayers() const { return numLayers_; }

    // f/w parameter

    // magnetic field f/w value in T
    double bField() const { return bField_; }

    // Analzyer

    // open and analyze TrackingParticles, original TTStubs and Association between them
    bool useMCTruth() const { return useMCTruth_; }
    // tag of AssociationMap between TTCluster and TrackingParticles
    const edm::InputTag& inputTagTTClusterAssMap() const { return inputTagTTClusterAssMap_; }
    // label of DTC producer
    const std::string& producerLabel() const { return producerLabel_; }

    // TP

    // pt cut in GeV
    double tpMinPt() const { return tpMinPt_; }
    // eta cut
    double tpMaxEta() const { return tpMaxEta_; }
    // cut on vertex pos r in cm
    double tpMaxVertR() const { return tpMaxVertR_; }
    // cut on vertex pos z in cm
    double tpMaxVertZ() const { return tpMaxVertZ_; }
    // cut on impact parameter in cm
    double tpMaxD0() const { return tpMaxD0_; }
    // required number of associated layers to a TP to consider it reconstruct-able
    int tpMinLayers() const { return tpMinLayers_; }
    // required number of associated ps layers to a TP to consider it reconstruct-able
    int tpMinLayersPS() const { return tpMinLayersPS_; }

    // derived Router parameter

    // total number of outer tracker DTCs
    int numDTCs() const { return numDTCs_; }
    // number of DTCs connected to one TFP (48)
    int numDTCsPerTFP() const { return numDTCsPerTFP_; }
    // total number of max possible outer tracker modules (72 per DTC)
    int numModules() const { return numModules_; }
    // number of inputs per systolic arrays in dtc firmware
    int numModulesPerRoutingBlock() const { return numModulesPerRoutingBlock_; }
    // max number of incomming stubs per packet (emp limit not cic limit)
    int maxFramesChannelInput() const { return maxFramesChannelInput_; }
    // max number out outgoing stubs per packet
    int maxFramesChannelOutput() const { return maxFramesChannelOutput_; }

    // derived Converter parameter

    // number of bits used for internal stub r - ChosenRofPhi
    int widthR() const { return widthR_; }
    // number of bits used for internal stub phi w.r.t. region centre
    int widthPhi() const { return widthPhi_; }
    // number of bits used for internal stub z
    int widthZ() const { return widthZ_; }
    // number of merged rows for look up
    int numMergedRows() const { return numMergedRows_; }
    // number of bits used for stub layer id
    int widthLayer() const { return widthLayer_; }
    // number of bits used for phi of row slope
    int widthM() const { return widthM_; }
    // number of bits used for phi or row intercept
    int widthC() const { return widthC_; }
    // number of bits used for internal stub eta
    int widthEta() const { return widthEta_; }
    // converts GeV in 1/cm
    double invPtToDphi() const { return invPtToDphi_; }
    // range of internal stub q over pt
    double rangeQoverPt() const { return rangeQoverPt_; }
    // cut on stub cot
    double maxCot() const { return maxCot_; }
    // cut on stub q over pt
    double maxQoverPt() const { return maxQoverPt_; }
    // region size in rad
    double baseRegion() const { return baseRegion_; }
    // internal stub q over pt precision in 1 /cm
    double baseQoverPt() const { return baseQoverPt_; }
    // internal stub r precision in cm
    double baseR() const { return baseR_; }
    // internal stub z precision in cm
    double baseZ() const { return baseZ_; }
    // internal stub phi precision in rad
    double basePhi() const { return basePhi_; }
    // phi of row slope precision in rad / pitch unit
    double baseM() const { return baseM_; }
    // phi of row intercept precision in rad
    double baseC() const { return baseC_; }

    // event setup

    const ::TrackerGeometry* trackerGeometry() const { return trackerGeometry_; }
    const ::TrackerTopology* trackerTopology() const { return trackerTopology_; }

    // derived event setup

    // check if TrackerGeometry is supported
    bool configurationSupported() const { return configurationSupported_; }

  private:
    // DataFormats

    std::unique_ptr<SettingsTMTT> tmtt_;
    std::unique_ptr<SettingsHybrid> hybrid_;

    //TrackerDTCProducer parameter sets

    const edm::ParameterSet paramsED_;
    const edm::ParameterSet paramsRouter_;
    const edm::ParameterSet paramsConverter_;
    const edm::ParameterSet paramsTracker_;
    const edm::ParameterSet paramsFW_;
    const edm::ParameterSet paramsFormat_;
    const edm::ParameterSet paramsAnalyzer_;
    const edm::ParameterSet paramsTP_;
    // ED parameter

    const edm::InputTag inputTagTTStubDetSetVec_;
    const edm::ESInputTag inputTagMagneticField_;
    const edm::ESInputTag inputTagTrackerGeometry_;
    const edm::ESInputTag inputTagTrackerTopology_;
    const edm::ESInputTag inputTagCablingMap_;
    const edm::ESInputTag inputTagTTStubAlgorithm_;
    const edm::ESInputTag inputTagGeometryConfiguration_;
    const std::string supportedTrackerXMLPSet_;
    const std::string supportedTrackerXMLPath_;
    const std::string supportedTrackerXMLFile_;
    const std::vector<std::string> supportedTrackerXMLVersions_;
    const std::string productBranchAccepted_;
    const std::string productBranchLost_;
    // "Hybrid" and "TMTT" format supported
    const std::string dataFormat_;
    // tk layout det id minus DetSetVec->detId
    const int offsetDetIdDSV_;
    // tk layout det id minus TrackerTopology lower det id
    const int offsetDetIdTP_;
    // offset in layer ids between barrel layer and endcap disks
    const int offsetLayerDisks_;
    // offset between 0 and smallest layer id (barrel layer 1)
    const int offsetLayerId_;
    const bool checkHistory_;
    const std::string processName_;
    const std::string productLabel_;

    // router parameter

    // enables emulation of truncation
    const double enableTruncation_;
    // Frequency in MHz, has to be integer multiple of FreqLHC
    const double freqDTC_;
    // time multiplexed period of track finding processor
    const int tmpTFP_;
    // needed gap between events of emp-infrastructure firmware
    const int numFramesInfra_;
    // number of systiloic arrays in stub router firmware
    const int numRoutingBlocks_;
    // fifo depth in stub router firmware
    const int sizeStack_;

    // converter parameter

    // number of row bits used in look up table
    const int widthRowLUT_;
    // number of bits used for stub qOverPt
    const int widthQoverPt_;

    // Tracker parameter

    // number of phi slices the outer tracker readout is organized in
    const int numRegions_;
    // number of regions a reconstructable particles may cross
    const int numOverlappingRegions_;
    // number of DTC boards used to readout a detector region
    const int numDTCsPerRegion_;
    // max number of sensor modules connected to one DTC board
    const int numModulesPerDTC_;
    // number of events collected in front-end
    const int tmpFE_;
    // number of bits used for internal stub bend
    const int widthBend_;
    // number of bits used for internal stub column
    const int widthCol_;
    // number of bits used for internal stub row
    const int widthRow_;
    // precision of internal stub bend in pitch units
    const double baseBend_;
    // precision of internal stub column in pitch units
    const double baseCol_;
    // precision of internal stub row in pitch units
    const double baseRow_;
    // used stub bend uncertainty in pitch units
    const double bendCut_;
    // LHC bunch crossing rate in MHz
    const double freqLHC_;

    // f/w constants

    // in e8 m/s
    const double speedOfLight_;
    // in T
    const double bField_;
    // accepted difference to EventSetup in T
    const double bFieldError_;
    // outer radius of outer tracker in cm
    const double outerRadius_;
    // inner radius of outer tracker in cm
    const double innerRadius_;
    // max strip/pixel pitch of outer tracker sensors in cm
    const double maxPitch_;

    // format specific parameter

    // cut on stub eta
    const double maxEta_;
    // cut on stub pt, also defines region overlap shape in GeV
    const double minPt_;
    // critical radius defining region overlap shape in cm
    const double chosenRofPhi_;
    // max number of detector layer connected to one DTC (hybrid) number of detector layers a reconstructbale particle may cross (tmtt)
    const int numLayers_;

    // Analzyer

    const bool useMCTruth_;
    const edm::InputTag inputTagTTClusterAssMap_;
    const std::string producerLabel_;

    // TP

    const double tpMinPt_;
    const double tpMaxEta_;
    const double tpMaxVertR_;
    const double tpMaxVertZ_;
    const double tpMaxD0_;
    const int tpMinLayers_;
    const int tpMinLayersPS_;

    // derived router parameter

    // total number of outer tracker DTCs
    int numDTCs_;
    // number of DTCs connected to one TFP (48)
    int numDTCsPerTFP_;
    // total number of max possible outer tracker modules (72 per DTC)
    int numModules_;
    // number of inputs per systolic arrays in dtc firmware
    int numModulesPerRoutingBlock_;
    // max number of incomming stubs per packet (emp limit not cic limit)
    int maxFramesChannelInput_;
    // max number out outgoing stubs per packet
    int maxFramesChannelOutput_;

    // derived Converter parameter

    // number of bits used for internal stub r - ChosenRofPhi
    int widthR_;
    // number of bits used for internal stub phi w.r.t. region centre
    int widthPhi_;
    // number of bits used for internal stub z
    int widthZ_;
    // number of merged rows for look up
    int numMergedRows_;
    // number of bits used for stub layer id
    int widthLayer_;
    // number of bits used for phi of row slope
    int widthM_;
    // number of bits used for phi or row intercept
    int widthC_;
    // number of bits used for internal stub eta
    int widthEta_;
    // converts GeV in 1/cm
    double invPtToDphi_;
    // range of internal stub q over pt
    double rangeQoverPt_;
    // cut on stub cot
    double maxCot_;
    // cut on stub q over pt
    double maxQoverPt_;
    // region size in rad
    double baseRegion_;
    // internal stub q over pt precision in 1 /cm
    double baseQoverPt_;
    // internal stub r precision in cm
    double baseR_;
    // internal stub z precision in cm
    double baseZ_;
    // internal stub phi precision in rad
    double basePhi_;
    // phi of row slope precision in rad / pitch unit
    double baseM_;
    // phi of row intercept precision in rad
    double baseC_;

    // event setup

    const ::TrackerGeometry* trackerGeometry_;
    const ::TrackerTopology* trackerTopology_;
    const ::MagneticField* magneticField_;
    const ::TrackerDetToDTCELinkCablingMap* ttCablingMap_;
    edm::ESHandle<TTStubAlgorithm<Ref_Phase2TrackerDigi_>> handleTTStubAlgorithm_;
    edm::ESHandle<DDCompactView> handleGeometryConfiguration_;
    edm::ProcessHistory processHistory_;

    // derived event setup

    // value = track trigger module id [0-15551], key = det id
    std::unordered_map<::DetId, int> cablingMap_;
    // collection of outer tracker sensor modules
    std::vector<Module> modules_;
    // collection of outer tracker sensor modules organised in DTCS [0-215][0-71]
    std::vector<std::vector<Module*>> dtcModules_;
    // true if tracker geometry and bfield is supported
    bool configurationSupported_;
  };

}  // namespace trackerDTC

#endif
