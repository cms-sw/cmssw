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
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackerDTC/interface/SettingsHybrid.h"
#include "L1Trigger/TrackerDTC/interface/SettingsTMTT.h"

#include <vector>
#include <memory>

namespace trackerDTC {

  class Module;

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
    // store TTStubAlgorithm handle
    void setCablingMap(const TrackerDetToDTCELinkCablingMap* cablingMap);
    // store ProcessHistory
    void setTTStubAlgorithm(const edm::ESHandle<TTStubAlgorithm<Ref_Phase2TrackerDigi_>>& handleTTStubAlgorithm);
    // store TrackerDetToDTCELinkCablingMap
    void setProcessHistory(const edm::ProcessHistory& processHistory);
    // check current coniguration consistency with input configuration
    void checkConfiguration();
    // convert cabling map
    void convertCablingMap();
    // store converted tracker geometry
    void setModules(std::vector<Module>& modules);
    // convert data fromat specific stuff
    void beginRun();
    // convert DetId to module id [0:15551]
    int modId(const ::DetId& detId) const;
    // collection of modules connected to a specific dtc
    const std::vector<Module*>& modules(int dtcId) const;

    // ED parameter

    edm::InputTag inputTagTTStubDetSetVec() const { return inputTagTTStubDetSetVec_; }
    edm::ESInputTag inputTagMagneticField() const { return inputTagMagneticField_; }
    edm::ESInputTag inputTagTrackerGeometry() const { return inputTagTrackerGeometry_; }
    edm::ESInputTag inputTagTrackerTopology() const { return inputTagTrackerTopology_; }
    edm::ESInputTag inputTagCablingMap() const { return inputTagCablingMap_; }
    edm::ESInputTag inputTagTTStubAlgorithm() const { return inputTagTTStubAlgorithm_; }
    std::string productBranch() const { return productBranch_; }
    std::string dataFormat() const { return dataFormat_; }
    int offsetDetIdDSV() const { return offsetDetIdDSV_; }
    int offsetDetIdTP() const { return offsetDetIdTP_; }
    SettingsHybrid* hybrid() const { return hybrid_.get(); }
    SettingsTMTT* tmtt() const { return tmtt_.get(); }
    int offsetLayerDisks() const { return offsetLayerDisks_; }
    int offsetLayerId() const { return offsetLayerId_; }

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

    const ::TrackerGeometry* trackerGeometry() const { return trackerGeometry_; }
    const ::TrackerTopology* trackerTopology() const { return trackerTopology_; }

    // derived event setup

    const std::vector<::DetId>& cablingMap() const { return cablingMap_; }

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

    // ED parameter

    const edm::InputTag inputTagTTStubDetSetVec_;
    const edm::ESInputTag inputTagMagneticField_;
    const edm::ESInputTag inputTagTrackerGeometry_;
    const edm::ESInputTag inputTagTrackerTopology_;
    const edm::ESInputTag inputTagCablingMap_;
    const edm::ESInputTag inputTagTTStubAlgorithm_;
    const std::string productBranch_;
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

    // derived router parameter

    // total number of outer tracker DTCs
    int numDTCs_;
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
    edm::ProcessHistory processHistory_;

    // derived event setup

    // index = track trigger module id [0-15551], value = det id
    std::vector<::DetId> cablingMap_;
    // collection of outer tracker sensor modules organised in DTCS [0-215][0-71]
    std::vector<std::vector<Module*>> dtcModules_;
  };

}  // namespace trackerDTC

#endif
