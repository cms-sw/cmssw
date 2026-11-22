#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackerDTC/interface/Setup.h"

#include <memory>
#include <vector>
#include <string>

namespace trackerDTC {

  /*! \class  trackerDTC::ProducerSetup
   *  \brief  Class to produce Setup providing constants, data formats and algorithms used by DTC emulator
   *  \author Thomas Schuh
   *  \date   2025, Dec
   */
  class ProducerSetup : public edm::ESProducer {
  public:
    ProducerSetup(const edm::ParameterSet& iConfig);
    ~ProducerSetup() override = default;
    std::unique_ptr<Setup> produce(const SetupRcd& rcd);

  private:
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeometry_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopology_;
    edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> getTokenCablingMap_;
    edm::ESGetToken<StubAlgorithm, TTStubAlgorithmRecord> getTokenTTStubAlgorithm_;
    Setup::Config config_;
  };

  ProducerSetup::ProducerSetup(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    getTokenTrackerGeometry_ = cc.consumes();
    getTokenTrackerTopology_ = cc.consumes();
    getTokenCablingMap_ = cc.consumes();
    getTokenTTStubAlgorithm_ = cc.consumes();
    const edm::ParameterSet& print = iConfig.getParameter<edm::ParameterSet>("Print");
    config_.printConstants = print.getParameter<bool>("Constants");
    config_.printEncodingBend = print.getParameter<bool>("EncodingBend");
    config_.printIDs = print.getParameter<std::vector<int>>("IDs");
    config_.printPath = print.getParameter<std::string>("Path");
    const edm::ParameterSet& cbc = iConfig.getParameter<edm::ParameterSet>("CBC");
    config_.cbcNumRow = cbc.getParameter<int>("NumRow");
    config_.cbcNumCol = cbc.getParameter<int>("NumCol");
    config_.cbcNumStub = cbc.getParameter<int>("NumStub");
    config_.cbcNumBX = cbc.getParameter<int>("NumBX");
    config_.cbcWidthBend = cbc.getParameter<int>("WidthBend");
    config_.cbcPitch = cbc.getParameter<double>("Pitch");
    config_.cbcLength = cbc.getParameter<double>("Length");
    const edm::ParameterSet& mpa = iConfig.getParameter<edm::ParameterSet>("MPA");
    config_.mpaNumRow = mpa.getParameter<int>("NumRow");
    config_.mpaNumCol = mpa.getParameter<int>("NumCol");
    config_.mpaNumStub = mpa.getParameter<int>("NumStub");
    config_.mpaNumBX = mpa.getParameter<int>("NumBX");
    config_.mpaWidthBend = mpa.getParameter<int>("WidthBend");
    config_.mpaPitch = mpa.getParameter<double>("Pitch");
    config_.mpaLength = mpa.getParameter<double>("Length");
    const edm::ParameterSet& cic = iConfig.getParameter<edm::ParameterSet>("CIC");
    config_.cicNumBX = cic.getParameter<int>("NumBX");
    config_.cicNumStub5g = cic.getParameter<int>("NumStub5g");
    config_.cicNumStub10g = cic.getParameter<int>("NumStub10g");
    config_.cicNumFEC = cic.getParameter<int>("NumFEC");
    const edm::ParameterSet& sm = iConfig.getParameter<edm::ParameterSet>("SensorModule");
    config_.smNumCIC = sm.getParameter<int>("NumCIC");
    config_.smTiltApproxSlope = sm.getParameter<double>("TiltApproxSlope");
    config_.smTiltApproxIntercept = sm.getParameter<double>("TiltApproxIntercept");
    config_.smTiltUncertaintyR = sm.getParameter<double>("TiltUncertaintyR");
    config_.smScattering = sm.getParameter<double>("Scattering");
    config_.smBendCut = sm.getParameter<double>("BendCut");
    config_.smClusterWidth = sm.getParameter<std::vector<double>>("ClusterWidth");
    config_.smAddPhiUncertainty = sm.getParameter<std::vector<double>>("AddPhiUncertainty");
    const edm::ParameterSet& dtc = iConfig.getParameter<edm::ParameterSet>("DTC");
    config_.dtcNumLayer = dtc.getParameter<int>("NumLayer");
    config_.dtcNumModule = dtc.getParameter<int>("NumModule");
    config_.dtcFreq = dtc.getParameter<double>("Freq");
    const edm::ParameterSet& reg = iConfig.getParameter<edm::ParameterSet>("Region");
    config_.regNumDTC = reg.getParameter<int>("NumDTC");
    config_.regNumTFP = reg.getParameter<int>("NumTFP");
    config_.regMinPt = reg.getParameter<double>("MinPt");
    config_.regMaxD0 = reg.getParameter<double>("MaxD0");
    config_.regBeamWindowZ = reg.getParameter<double>("BeamWindowZ");
    config_.regMaxEta = reg.getParameter<double>("MaxEta");
    config_.regChosenRofPhi = reg.getParameter<double>("ChosenRofPhi");
    config_.regChosenRofZ = reg.getParameter<double>("ChosenRofZ");
    const edm::ParameterSet& sys = iConfig.getParameter<edm::ParameterSet>("System");
    config_.sysNumModule = sys.getParameter<int>("NumModule");
    config_.sysNumRegion = sys.getParameter<int>("NumRegion");
    config_.sysNumOverlap = sys.getParameter<int>("NumOverlap");
    config_.sysNumATCASlot = sys.getParameter<int>("NumATCASlot");
    config_.sysSlotLimitPS = sys.getParameter<int>("SlotLimitPS");
    config_.sysSlotLimit10gbps = sys.getParameter<int>("SlotLimit10gbps");
    config_.sysNumBarrelLayer = sys.getParameter<int>("NumBarrelLayer");
    config_.sysNumBarrelLayerPS = sys.getParameter<int>("NumBarrelLayerPS");
    config_.sysNumFramesInfra = sys.getParameter<int>("NumFramesInfra");
    config_.sysNumLayer = sys.getParameter<int>("NumLayers");
    config_.sysSpeedOfLight = sys.getParameter<double>("SpeedOfLight");
    config_.sysBField = sys.getParameter<double>("BField");
    config_.sysOuterRadius = sys.getParameter<double>("OuterRadius");
    config_.sysInnerRadius = sys.getParameter<double>("InnerRadius");
    config_.sysHalfLength = sys.getParameter<double>("HalfLength");
    config_.sysLhcFreq = sys.getParameter<double>("FreqLHC");
    const edm::ParameterSet& fe = iConfig.getParameter<edm::ParameterSet>("StubFE");
    config_.feBaseBend = fe.getParameter<double>("BaseBend");
    config_.feBaseCol = fe.getParameter<double>("BaseCol");
    config_.feBaseRow = fe.getParameter<double>("BaseRow");
    const edm::ParameterSet& gl = iConfig.getParameter<edm::ParameterSet>("StubGL");
    config_.glWidthR = gl.getParameter<int>("WidthR");
    config_.glWidthPhi = gl.getParameter<int>("WidthPhi");
    config_.glWidthZ = gl.getParameter<int>("WidthZ");
    const edm::ParameterSet& stub = iConfig.getParameter<edm::ParameterSet>("StubDTC");
    config_.stubNumRingsPS = stub.getParameter<std::vector<int>>("NumRingsPS");
    config_.stubLayerRs = stub.getParameter<std::vector<double>>("LayerRs");
    config_.stubDiskZs = stub.getParameter<std::vector<double>>("DiskZs");
    const std::vector<edm::ParameterSet> disk2SRsSet = stub.getParameter<std::vector<edm::ParameterSet>>("Disk2SRsSet");
    config_.stubDisk2SRs.reserve(disk2SRsSet.size());
    for (const auto& pSet : disk2SRsSet)
      config_.stubDisk2SRs.emplace_back(pSet.getParameter<std::vector<double>>("Disk2SRs"));
    config_.stubWidthsND = stub.getParameter<std::vector<int>>("WidthsND");
    config_.stubWidthsR = stub.getParameter<std::vector<int>>("WidthsR");
    config_.stubWidthsZ = stub.getParameter<std::vector<int>>("WidthsZ");
    config_.stubWidthsPhi = stub.getParameter<std::vector<int>>("WidthsPhi");
    config_.stubWidthsAlpha = stub.getParameter<std::vector<int>>("WidthsAlpha");
    config_.stubWidthsBend = stub.getParameter<std::vector<int>>("WidthsBend");
    config_.stubRangesR = stub.getParameter<std::vector<double>>("RangesR");
    config_.stubRangesZ = stub.getParameter<std::vector<double>>("RangesZ");
    config_.stubRangesAlpha = stub.getParameter<std::vector<double>>("RangesAlpha");
    config_.stubOffsetRDiskPS = stub.getParameter<double>("OffsetRDiskPS");
    config_.stubMinPt = stub.getParameter<double>("MinPt");
    const edm::ParameterSet& un = iConfig.getParameter<edm::ParameterSet>("Unbox");
    config_.unWidthAddr = un.getParameter<int>("WidthAddr");
    config_.unNumNode = un.getParameter<int>("NumNode");
    const edm::ParameterSet& re = iConfig.getParameter<edm::ParameterSet>("Repack");
    config_.reIn = re.getParameter<int>("In");
    config_.reOut = re.getParameter<int>("Out");
    const edm::ParameterSet& fw = iConfig.getParameter<edm::ParameterSet>("Firmware");
    config_.fwEnableTruncation = fw.getParameter<bool>("EnableTruncation");
    config_.fwWidthDSPa = fw.getParameter<int>("WidthDSPa");
    config_.fwWidthDSPb = fw.getParameter<int>("WidthDSPb");
    config_.fwWidthDSPc = fw.getParameter<int>("WidthDSPc");
    config_.fwWidthAddrBRAM36 = fw.getParameter<int>("WidthAddrBRAM36");
    config_.fwWidthAddrBRAM18 = fw.getParameter<int>("WidthAddrBRAM18");
  }

  std::unique_ptr<Setup> ProducerSetup::produce(const SetupRcd& rcd) {
    const TrackerGeometry& trackerGeometry = rcd.get(getTokenTrackerGeometry_);
    const TrackerTopology& trackerTopology = rcd.get(getTokenTrackerTopology_);
    const TrackerDetToDTCELinkCablingMap& cablingMap = rcd.get(getTokenCablingMap_);
    const edm::ESHandle<StubAlgorithm> handleStubAlgorithm = rcd.getHandle(getTokenTTStubAlgorithm_);
    const StubAlgorithmOfficial& stubAlgoritm =
        *dynamic_cast<const StubAlgorithmOfficial*>(&rcd.get(getTokenTTStubAlgorithm_));
    const edm::ParameterSet& pSet = getParameterSet(handleStubAlgorithm.description()->pid_);
    Setup::ConfigTTStubAlgorithm configTTStubAlgorithm;
    configTTStubAlgorithm.nTiltedRings = pSet.getParameter<std::vector<double>>("NTiltedRings");
    configTTStubAlgorithm.barrelCut = pSet.getParameter<std::vector<double>>("BarrelCut");
    const auto& pSetsTiltedLayer = pSet.getParameter<std::vector<edm::ParameterSet>>("TiltedBarrelCutSet");
    const auto& pSetsEncapDisks = pSet.getParameter<std::vector<edm::ParameterSet>>("EndcapCutSet");
    configTTStubAlgorithm.tiltedBarrelCutSet.reserve(pSetsTiltedLayer.size());
    for (const auto& pSet : pSetsTiltedLayer)
      configTTStubAlgorithm.tiltedBarrelCutSet.emplace_back(pSet.getParameter<std::vector<double>>("TiltedCut"));
    configTTStubAlgorithm.endcapCutSet.reserve(pSetsEncapDisks.size());
    for (const auto& pSet : pSetsEncapDisks)
      configTTStubAlgorithm.endcapCutSet.emplace_back(pSet.getParameter<std::vector<double>>("EndcapCut"));
    return std::make_unique<Setup>(
        config_, trackerGeometry, trackerTopology, cablingMap, stubAlgoritm, configTTStubAlgorithm);
  }

}  // namespace trackerDTC

DEFINE_FWK_EVENTSETUP_MODULE(trackerDTC::ProducerSetup);
