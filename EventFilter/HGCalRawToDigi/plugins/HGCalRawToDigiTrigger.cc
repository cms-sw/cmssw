#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONDPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalECONTPacketInfoHost.h"
#include "DataFormats/HGCalDigi/interface/HGCalFEDPacketInfoHost.h"
#include "CondFormats/DataRecord/interface/HGCalModuleConfigurationRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalTriggerConfiguration.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiTriggerHost.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpackerTrigger.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexerTrigger.h"


class HGCalRawToDigiTrigger : public edm::stream::EDProducer<edm::stream::WatchRuns> {

public:
  explicit HGCalRawToDigiTrigger(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  // input tokens
  const edm::EDGetTokenT<RawDataBuffer> fedRawTriggerToken_;

  // output tokens
  const edm::EDPutTokenT<hgcaldigi::HGCalDigiTriggerHost> digisTriggerToken_;
  const edm::EDPutTokenT<hgcaldigi::HGCalECONTPacketInfoHost> econtPacketInfoToken_;
  const edm::EDPutTokenT<hgcaldigi::HGCalFEDTriggerPacketInfoHost> fedTriggerPacketInfoToken_;

  // config tokens and objects
  const edm::ESGetToken<HGCalMappingModuleIndexerTrigger, HGCalElectronicsMappingRcd> moduleIndexToken_;
  const edm::ESGetToken<HGCalTriggerConfiguration, HGCalModuleConfigurationRcd> configToken_;
  HGCalUnpackerTrigger unpacker_trigger_;
};

HGCalRawToDigiTrigger::HGCalRawToDigiTrigger(const edm::ParameterSet& iConfig)
    : fedRawTriggerToken_(consumes<RawDataBuffer>(iConfig.getParameter<edm::InputTag>("src"))),
      digisTriggerToken_(produces<hgcaldigi::HGCalDigiTriggerHost>()),
      econtPacketInfoToken_(produces<hgcaldigi::HGCalECONTPacketInfoHost>()),
      fedTriggerPacketInfoToken_(produces<hgcaldigi::HGCalFEDTriggerPacketInfoHost>()),
      moduleIndexToken_(esConsumes()),
      configToken_(esConsumes()) {}

void HGCalRawToDigiTrigger::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
}

void HGCalRawToDigiTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& moduleIndexer = iSetup.getData(moduleIndexToken_);
  //const auto& cellIndexer = iSetup.getData(cellIndexToken_);
  const auto& config = iSetup.getData(configToken_);
  
  hgcaldigi::HGCalDigiTriggerHost digisTrigger(cms::alpakatools::host(), moduleIndexer.maxDataSize());
  hgcaldigi::HGCalFEDTriggerPacketInfoHost fedTriggerPacketInfo(cms::alpakatools::host(), moduleIndexer.fedCount());
  hgcaldigi::HGCalECONTPacketInfoHost econtPacketInfo(cms::alpakatools::host(), moduleIndexer.maxModulesIndex());
  const auto& fedBuffer = iEvent.get(fedRawTriggerToken_);

  for (int32_t i = 0; i < digisTrigger.view().metadata().size(); i++) {
    for (int32_t ibx = 0; ibx < 7; ibx++) digisTrigger.view()[i].valid()(ibx,0) = false;
    digisTrigger.view()[i].algo() = 0;
  }
  LogDebug("HGCalRawToDigiTrigger::produce") << " packet info metadata size " << econtPacketInfo.view().metadata().size();
  for (int32_t i = 0; i < econtPacketInfo.view().metadata().size(); i++) {
    econtPacketInfo.view()[i].exception() = 0;
    econtPacketInfo.view()[i].location() = 0;
    econtPacketInfo.view()[i].payloadLength() = 0;
    for (int32_t itc = 0; itc < 48; itc++) econtPacketInfo.view()[i].TCEnergy_Stage1()(0,itc) = 0xffff;
  }

  for(const auto& frs :  moduleIndexer.fedReadoutSequences() ) {
    if (frs.readoutTypes_.empty()) {
      continue;
    }
    auto fedId = frs.id;
    const auto& fed_data = fedBuffer.fragmentData(fedId);
    if (fed_data.size() == 0){
      continue;
    }
    
    const auto& fedConfig = config.feds[fedId];
    LogDebug("HGCalRawToDigiTrigger::produce") << "HGCalRawToDigiTrigger::produce - got a fed_data fragment size=" << fed_data.size() << " for fedid=" << fedId;
    LogDebug("HGCalRawToDigiTrigger::produce") << fedConfig.tdaqs.size() << " tdaq blocks and " << fedConfig.econtSwapOffset.size() << " ECONTs";
    
    //pedro : comment the unpacker should receive direcly fedConfig and not the full config, and maybe the fed readout sequence instead of the module indexer
    unpacker_trigger_.parseFEDData(fedId,fed_data,config,moduleIndexer,digisTrigger,fedTriggerPacketInfo,econtPacketInfo);
  }


  // put information to the event
  iEvent.emplace(digisTriggerToken_, std::move(digisTrigger));
  iEvent.emplace(econtPacketInfoToken_, std::move(econtPacketInfo));
  iEvent.emplace(fedTriggerPacketInfoToken_, std::move(fedTriggerPacketInfo));
}

// fill descriptions
void HGCalRawToDigiTrigger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  descriptions.add("hgcalDigisTrigger", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalRawToDigiTrigger);

