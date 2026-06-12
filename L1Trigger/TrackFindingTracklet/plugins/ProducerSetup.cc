#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackFindingTracklet/interface/Setup.h"

#include <memory>

namespace trklet {

  /*! \class  trklet::ProducerSetup
   *  \brief  Class to produce setup of Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class ProducerSetup : public edm::ESProducer {
  public:
    ProducerSetup(const edm::ParameterSet& iConfig);
    ~ProducerSetup() override = default;
    std::unique_ptr<Setup> produce(const trackerDTC::SetupRcd& setupRcd);

  private:
    Setup::Config config_;
    edm::ESGetToken<trackerDTC::Setup, trackerDTC::SetupRcd> esGetToken_;
  };

  ProducerSetup::ProducerSetup(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    esGetToken_ = cc.consumes();
    config_.enableTruncation = iConfig.getParameter<bool>("EnableTruncation");
    const edm::ParameterSet& pSetOT = iConfig.getParameter<edm::ParameterSet>("OT");
    config_.otLimitPSBarrel = pSetOT.getParameter<double>("LimitPSBarrel");
    config_.otLimitsTiltedR = pSetOT.getParameter<std::vector<double>>("LimitsTiltedR");
    config_.otLimitsTiltedZ = pSetOT.getParameter<std::vector<double>>("LimitsTiltedZ");
    config_.otLimitsPSDiksZ = pSetOT.getParameter<std::vector<double>>("LimitsPSDiksZ");
    config_.otLimitsPSDiskR = pSetOT.getParameter<std::vector<double>>("LimitsPSDiskR");
    const edm::ParameterSet& pSetIR = iConfig.getParameter<edm::ParameterSet>("IR");
    config_.irChannelsIn = pSetIR.getParameter<std::vector<int>>("ChannelsIn");
    const edm::ParameterSet& pSetTB = iConfig.getParameter<edm::ParameterSet>("TB");
    config_.tbFreq = pSetTB.getParameter<double>("Freq");
    config_.tbMinZ = pSetTB.getParameter<double>("MinZ");
    config_.tbMaxR = pSetTB.getParameter<double>("MaxR");
    config_.tbInnerRadius = pSetTB.getParameter<double>("InnerRadius");
    config_.tbNumSeedTypes = pSetTB.getParameter<int>("NumSeedTypes");
    config_.tbNumSeedingLayers = pSetTB.getParameter<int>("NumSeedingLayers");
    config_.tbNumLayers = pSetTB.getParameter<int>("NumLayers");
    config_.tbSeedTypes = pSetTB.getParameter<std::vector<std::string>>("SeedTypes");
    config_.tbSeedTypesSeedLayers.reserve(config_.tbNumSeedTypes);
    const edm::ParameterSet& stsl = pSetTB.getParameter<edm::ParameterSet>("SeedTypesSeedLayers");
    for (const std::string& seedType : config_.tbSeedTypes)
      config_.tbSeedTypesSeedLayers.emplace_back(stsl.getParameter<std::vector<int>>(seedType));
    config_.tbSeedTypesProjectionLayers.reserve(config_.tbNumSeedTypes);
    const edm::ParameterSet& stpl = pSetTB.getParameter<edm::ParameterSet>("SeedTypesProjectionLayers");
    for (const std::string& seedType : config_.tbSeedTypes)
      config_.tbSeedTypesProjectionLayers.emplace_back(stpl.getParameter<std::vector<int>>(seedType));
    config_.tbWidthsR = pSetTB.getParameter<std::vector<int>>("WidthsR");
    config_.tbWidthStubId = pSetTB.getParameter<int>("WidthStubId");
    config_.tbWidthInv2R = pSetTB.getParameter<int>("WidthInv2R");
    config_.tbWidthPhi0 = pSetTB.getParameter<int>("WidthPhi0");
    config_.tbWidthZ0 = pSetTB.getParameter<int>("WidthZ0");
    config_.tbWidthCot = pSetTB.getParameter<int>("WidthCot");
    const edm::ParameterSet& pSetTM = iConfig.getParameter<edm::ParameterSet>("TM");
    config_.tmMuxOrder = pSetTM.getParameter<std::vector<std::string>>("MuxOrder");
    const edm::ParameterSet& pSetDR = iConfig.getParameter<edm::ParameterSet>("DR");
    config_.drUseDTCStubs = pSetDR.getParameter<bool>("UseDTCStubs");
    config_.drUseTTStubs = pSetDR.getParameter<bool>("UseTTStubs");
    config_.drNumComparisonModules = pSetDR.getParameter<int>("NumComparisonModules");
    config_.drMinIdenticalStubs = pSetDR.getParameter<int>("MinIdenticalStubs");
    config_.drWidthR = pSetDR.getParameter<int>("WidthR");
    config_.drWidthPhi = pSetDR.getParameter<int>("WidthPhi");
    config_.drWidthZ = pSetDR.getParameter<int>("WidthZ");
    config_.drWidthDPhi = pSetDR.getParameter<int>("WidthDPhi");
    config_.drWidthDZ = pSetDR.getParameter<int>("WidthDZ");
    config_.drBaseShiftDPhi = pSetDR.getParameter<int>("BaseShiftDPhi");
    config_.drBaseShiftDZ = pSetDR.getParameter<int>("BaseShiftDZ");
    const edm::ParameterSet& pSetKF = iConfig.getParameter<edm::ParameterSet>("KF");
    config_.kfUseSimulation = pSetKF.getParameter<bool>("UseSimulation");
    config_.kfMaxTracks = pSetKF.getParameter<int>("MaxTracks");
    config_.kfNumLayers = pSetKF.getParameter<int>("NumLayers");
    config_.kfMinLayers = pSetKF.getParameter<int>("MinLayers");
    config_.kfBaseShiftPhi = pSetKF.getParameter<int>("BaseShiftPhi");
    config_.kfBaseShiftZ = pSetKF.getParameter<int>("BaseShiftZ");
    const edm::ParameterSet& pSetTQ = iConfig.getParameter<edm::ParameterSet>("TQ");
    config_.tqNumChannel = pSetTQ.getParameter<int>("NumChannel");
    config_.tqWidthChi21 = pSetTQ.getParameter<int>("WidthChi21");
    config_.tqWidthChi20 = pSetTQ.getParameter<int>("WidthChi20");
    config_.tqBaseShiftChi21 = pSetTQ.getParameter<int>("BaseShiftChi21");
    config_.tqBaseShiftChi20 = pSetTQ.getParameter<int>("BaseShiftChi20");
    config_.tqWidthInvV0 = pSetTQ.getParameter<int>("WidthInvV0");
    config_.tqWidthInvV1 = pSetTQ.getParameter<int>("WidthInvV1");
    config_.tqWidthMVA = pSetTQ.getParameter<int>("WidthMVA");
    config_.tqBinEdges = pSetTQ.getParameter<std::vector<int>>("BinEdges");
    const edm::ParameterSet& pSetTFP = iConfig.getParameter<edm::ParameterSet>("TFP");
    config_.tfpWidthPhi0 = pSetTFP.getParameter<int>("WidthPhi0");
    config_.tfpWidthInvR = pSetTFP.getParameter<int>("WidthInvR");
    config_.tfpWidthCot = pSetTFP.getParameter<int>("WidthCot");
    config_.tfpWidthZ0 = pSetTFP.getParameter<int>("WidthZ0");
    config_.tfpNumChannel = pSetTFP.getParameter<int>("NumChannel");
  }

  std::unique_ptr<Setup> ProducerSetup::produce(const trackerDTC::SetupRcd& setupRcd) {
    const trackerDTC::Setup* setup = &setupRcd.get(esGetToken_);
    return std::make_unique<Setup>(config_, setup);
  }
}  // namespace trklet

DEFINE_FWK_EVENTSETUP_MODULE(trklet::ProducerSetup);
