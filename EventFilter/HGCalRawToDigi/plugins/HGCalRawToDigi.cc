#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"

#include "CondFormats/DataRecord/interface/HGCalElectronicsMappingRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalConfiguration.h"

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb.h"

#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"
class HGCalRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit HGCalRawToDigi(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  // input tokens
  const edm::EDGetTokenT<FEDRawDataCollection> fedRawToken_;

  // output tokens
  const edm::EDPutTokenT<hgcaldigi::HGCalDigiHost> digisToken_;
  const edm::EDPutTokenT<hgcaldigi::HGCalECONDPacketInfoHost> econdPacketInfoToken_;

  // TODO @hqucms
  // what else do we want to output?

  // config tokens and objects
  edm::ESWatcher<HGCalElectronicsMappingRcd> mapWatcher_;
  edm::ESGetToken<HGCalMappingCellIndexer, HGCalElectronicsMappingRcd> cellIndexToken_;
  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalElectronicsMappingRcd> moduleIndexToken_;
  edm::ESGetToken<HGCalConfiguration, HGCalModuleConfigurationRcd> configToken_;
  HGCalMappingCellIndexer cellIndexer_;
  HGCalMappingModuleIndexer moduleIndexer_;
  HGCalConfiguration config_;

  // TODO @hqucms
  // how to implement this enabled eRx pattern? Can this be taken from the logical mapping?
  // HGCalCondSerializableModuleInfo::ERxBitPatternMap erxEnableBits_;
  // std::map<uint16_t, uint16_t> fed2slink_;

  // TODO @hqucms
  // HGCalUnpackerConfig unpackerConfig_;
  HGCalUnpacker unpacker_;

  const bool fixCalibChannel_;
};

HGCalRawToDigi::HGCalRawToDigi(const edm::ParameterSet& iConfig)
    : fedRawToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      digisToken_(produces<hgcaldigi::HGCalDigiHost>()),
      econdPacketInfoToken_(produces<hgcaldigi::HGCalECONDPacketInfoHost>()),
      cellIndexToken_(esConsumes<edm::Transition::BeginRun>()),
      moduleIndexToken_(esConsumes<edm::Transition::BeginRun>()),
      configToken_(esConsumes<edm::Transition::BeginRun>()),
      // unpackerConfig_(HGCalUnpackerConfig{.sLinkBOE = iConfig.getParameter<unsigned int>("slinkBOE"),
      //                                     .cbHeaderMarker = iConfig.getParameter<unsigned int>("cbHeaderMarker"),
      //                                     .econdHeaderMarker = iConfig.getParameter<unsigned int>("econdHeaderMarker"),
      //                                     .payloadLengthMax = iConfig.getParameter<unsigned int>("payloadLengthMax"),
      //                                     .applyFWworkaround = iConfig.getParameter<bool>("applyFWworkaround")}),
      fixCalibChannel_(iConfig.getParameter<bool>("fixCalibChannel")) {}

void HGCalRawToDigi::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  // retrieve logical mapping
  if (mapWatcher_.check(iSetup)) {
    moduleIndexer_ = iSetup.getData(moduleIndexToken_);
    cellIndexer_ = iSetup.getData(cellIndexToken_);
    config_ = iSetup.getData(configToken_);
  }

  // TODO @hqucms
  // retrieve configs: TODO
  // auto moduleInfo = iSetup.getData(moduleInfoToken_);

  // TODO @hqucms
  // init unpacker with proper configs
}

void HGCalRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  hgcaldigi::HGCalDigiHost digis(moduleIndexer_.getMaxDataSize(), cms::alpakatools::host());
  hgcaldigi::HGCalECONDPacketInfoHost econdPacketInfo(moduleIndexer_.getMaxModuleSize(), cms::alpakatools::host());
  // std::cout << "Created DIGIs SOA with " << digis.view().metadata().size() << " entries" << std::endl;

  // TODO @hqucms

  // retrieve the FED raw data
  const auto& raw_data = iEvent.get(fedRawToken_);

  for (int32_t i = 0; i < digis.view().metadata().size(); i++) {
    digis.view()[i].flags() = hgcal::DIGI_FLAG::NotAvailable;
  }
  // TODO: comparing timing of multithread and FED-level parrallelization
  // for (unsigned fedId = 0; fedId < moduleIndexer_.fedCount(); ++fedId) {
  //   const auto& fed_data = raw_data.FEDData(fedId);
  //   if (fed_data.size() == 0)
  //     continue;
  //   unpacker_.parseFEDData(fedId, fed_data, moduleIndexer_, config_, digis, econdPacketInfo, /*headerOnlyMode*/ false);
  // }

  //Parallelization
  tbb::this_task_arena::isolate([&]() {
    tbb::parallel_for(0U, moduleIndexer_.fedCount(), [&](unsigned fedId) {
      const auto& fed_data = raw_data.FEDData(fedId);
      if (fed_data.size() == 0)
        return;
      unpacker_.parseFEDData(
          fedId, fed_data, moduleIndexer_, config_, digis, econdPacketInfo, /*headerOnlyMode*/ false);
      return;
    });
  });
  // put information to the event
  iEvent.emplace(digisToken_, std::move(digis));
  iEvent.emplace(econdPacketInfoToken_, std::move(econdPacketInfo));
}

// fill descriptions
void HGCalRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int> >("fedIds", {});
  desc.add<bool>("fixCalibChannel", true)
      ->setComment("FIXME: always treat calib channels in characterization mode; to be fixed in ROCv3b");
  descriptions.add("hgcalDigis", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalRawToDigi);
