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

#include "CondFormats/DataRecord/interface/HGCalMappingModuleIndexerRcd.h"
// #include "CondFormats/DataRecord/interface/HGCalMappingModuleRcd.h"
#include "CondFormats/DataRecord/interface/HGCalMappingCellIndexerRcd.h"
// #include "CondFormats/DataRecord/interface/HGCalMappingCellRcd.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"
// #include "CondFormats/HGCalObjects/interface/alpaka/HGCalMappingParameterDeviceCollection.h"
// #include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

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

  // TODO @hqucms
  // what else do we want to output?

  // const edm::EDPutTokenT<HGCalFlaggedECONDInfoCollection> flaggedRawDataToken_;
  // const edm::EDPutTokenT<HGCalElecDigiCollection> elecDigisToken_;
  // const edm::EDPutTokenT<HGCalElecDigiCollection> elecCMsToken_;

  // config tokens and objects
  edm::ESWatcher<HGCalMappingModuleIndexerRcd> mapWatcher_;
  edm::ESGetToken<HGCalMappingCellIndexer, HGCalMappingCellIndexerRcd> cellIndexToken_;
  edm::ESGetToken<HGCalMappingModuleIndexer, HGCalMappingModuleIndexerRcd> moduleIndexToken_;
  HGCalMappingCellIndexer cellIndexer_;
  HGCalMappingModuleIndexer moduleIndexer_;

  // TODO @hqucms
  // how to implement this enabled eRx pattern? Can this be taken from the logical mapping?
  // HGCalCondSerializableModuleInfo::ERxBitPatternMap erxEnableBits_;
  // std::map<uint16_t, uint16_t> fed2slink_;

  // TODO @hqucms
  // HGCalUnpackerConfig unpackerConfig_;
  // HGCalCondSerializableConfig config_;
  HGCalUnpacker unpacker_;

  const bool fixCalibChannel_;
};

HGCalRawToDigi::HGCalRawToDigi(const edm::ParameterSet& iConfig)
    : fedRawToken_(consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      digisToken_(produces<hgcaldigi::HGCalDigiHost>()),
      // flaggedRawDataToken_(produces<HGCalFlaggedECONDInfoCollection>("UnpackerFlags")),
      // elecDigisToken_(produces<HGCalElecDigiCollection>("DIGI")),
      // elecCMsToken_(produces<HGCalElecDigiCollection>("CM")),
      cellIndexToken_(esConsumes<edm::Transition::BeginRun>()),
      moduleIndexToken_(esConsumes<edm::Transition::BeginRun>()),
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
  }

  // TODO @hqucms
  // retrieve configs: TODO
  // auto moduleInfo = iSetup.getData(moduleInfoToken_);

  // TODO @hqucms
  // init unpacker with proper configs
}

void HGCalRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  hgcaldigi::HGCalDigiHost digis(cellIndexer_.maxDenseIndex(), cms::alpakatools::host());
  // std::cout << "Created DIGIs SOA with " << digis.view().metadata().size() << " entries" << std::endl;

  // TODO @hqucms
  // CM and error flags output
  hgcaldigi::HGCalDigiHost common_modes(cellIndexer_.maxDenseIndex(), cms::alpakatools::host());
  std::vector<HGCalFlaggedECONDInfo> errors;

  // retrieve the FED raw data
  const auto& raw_data = iEvent.get(fedRawToken_);

  for (unsigned fedId = 0; fedId < moduleIndexer_.nfeds_; ++fedId) {
    const auto& fed_data = raw_data.FEDData(fedId);
    if (fed_data.size() == 0)
      continue;
    unpacker_.parseFEDData(fedId, fed_data, digis, common_modes, errors);
  }

  // TODO @hqucms
  //   try {
  //     unpacker_->parseSLink(
  //         data_32bit,
  //         [this](uint16_t sLink, uint8_t captureBlock, uint8_t econd) {
  //           return this->erxEnableBits_[HGCalCondSerializableModuleInfo::erxBitPatternMapDenseIndex(
  //               sLink, captureBlock, econd, 0, 0)];
  //         },
  //         [this](uint16_t fedId) {
  //           if (auto it = this->fed2slink_.find(fedId); it != this->fed2slink_.end()) {
  //             return this->fed2slink_.at(fedId);
  //           } else {
  //             // FIXME: throw an error or return 0?
  //             return (uint16_t)0;
  //           }
  //         });
  //   } catch (cms::Exception& e) {
  //     std::cout << "An exeption was caught while decoding raw data for FED " << std::dec << (uint32_t)fedId
  //               << std::endl;
  //     std::cout << e.what() << std::endl;
  //     std::cout << "Event is: " << std::endl;
  //     std::cout << "Total size (32b words) " << std::dec << data_32bit.size() << std::endl;
  //     for (size_t i = 0; i < data_32bit.size(); i++)
  //       std::cout << std::dec << i << " | " << std::hex << "0x" << std::setfill('0') << data_32bit[i] << std::endl;
  //   }

  //   auto channeldata = unpacker_->channelData();
  //   auto commonModeSum = unpacker_->commonModeSum();
  //   for (unsigned int i = 0; i < channeldata.size(); i++) {
  //     auto data = channeldata.at(i);
  //     const auto& id = data.id();
  //     auto idraw = id.raw();
  //     auto raw = data.raw();
  //     LogDebug("HGCalRawToDigi::produce") << "channel data, id=" << idraw << ", raw=" << raw;
  //     elec_digis.push_back(data);
  //     elecid.push_back(id.raw());
  //     tctp.push_back(data.tctp());
  //     uint32_t modid = id.econdIdxRawId();  // remove first 10 bits to get full electronics ID of ECON-D module
  //     // FIXME: in the current HGCROC the calib channels (=18) is always in characterization model; to be fixed in ROCv3b
  //     auto charMode = config_.moduleConfigs[modid].charMode || (fixCalibChannel_ && id.halfrocChannel() == 18);
  //     adcm1.push_back(data.adcm1(charMode));
  //     adc.push_back(data.adc(charMode));
  //     tot.push_back(data.tot(charMode));
  //     toa.push_back(data.toa());
  //     cm.push_back(commonModeSum.at(i));
  //   }

  //   auto commonmode = unpacker_->commonModeData();
  //   for (unsigned int i = 0; i < commonmode.size(); i++) {
  //     auto cm = commonmode.at(i);
  //     const auto& id = cm.id();
  //     auto idraw = id.raw();
  //     auto raw = cm.raw();
  //     LogDebug("HGCalRawToDigi::produce") << "common modes, id=" << idraw << ", raw=" << raw;
  //     elec_cms.push_back(cm);
  //   }

  //   // append flagged ECONDs
  //   flagged_econds.insert(flagged_econds.end(), unpacker_->flaggedECOND().begin(), unpacker_->flaggedECOND().end());
  // }

  // // check how many flagged ECOND-s we have
  // if (!flagged_econds.empty()) {
  //   LogDebug("HGCalRawToDigi:produce") << " caught " << flagged_econds.size() << " ECON-D with poor quality flags";
  //   if (flagged_econds.size() > flaggedECONDMax_) {
  //     throw cms::Exception("HGCalRawToDigi:produce")
  //         << "Too many flagged ECON-Ds: " << flagged_econds.size() << " > " << flaggedECONDMax_ << ".";
  //   }
  // }

  // TODO @hqucms
  // fill dummy outputs
  for (unsigned int i = 0; i < cellIndexer_.maxDenseIndex(); i++) {
    digis.view()[i].tctp() = 0;
    digis.view()[i].adcm1() = 0;
    digis.view()[i].adc() = 0;
    digis.view()[i].tot() = 0;
    digis.view()[i].toa() = 0;
    digis.view()[i].cm() = 0;
    digis.view()[i].flags() = 0;
  }

  // put information to the event
  iEvent.emplace(digisToken_, std::move(digis));
  // iEvent.emplace(flaggedRawDataToken_, std::move(flagged_econds));
  // iEvent.emplace(elecDigisToken_, std::move(elec_digis));
  // iEvent.emplace(elecCMsToken_, std::move(elec_cms));
}

// fill descriptions
void HGCalRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("rawDataCollector"));
  desc.add<std::vector<unsigned int> >("fedIds", {});
  // desc.add<unsigned int>("maxCaptureBlock", 1)->setComment("maximum number of capture blocks in one S-Link");
  // desc.add<unsigned int>("cbHeaderMarker", 0x5f)->setComment("capture block reserved pattern");
  // desc.add<unsigned int>("econdHeaderMarker", 0x154)->setComment("ECON-D header Marker pattern");
  // desc.add<unsigned int>("slinkBOE", 0x55)->setComment("SLink BOE pattern");
  // desc.add<unsigned int>("captureBlockECONDMax", 12)->setComment("maximum number of ECON-Ds in one capture block");
  // desc.add<bool>("applyFWworkaround", false)
  //     ->setComment("use to enable dealing with firmware features (e.g. repeated words)");
  // desc.add<bool>("swap32bendianness", false)->setComment("use to swap 32b endianness");
  // desc.add<unsigned int>("econdERXMax", 12)->setComment("maximum number of eRxs in one ECON-D");
  // desc.add<unsigned int>("erxChannelMax", 37)->setComment("maximum number of channels in one eRx");
  // desc.add<unsigned int>("payloadLengthMax", 469)->setComment("maximum length of payload length");
  // desc.add<unsigned int>("channelMax", 7000000)->setComment("maximum number of channels unpacked");
  // desc.add<unsigned int>("commonModeMax", 4000000)->setComment("maximum number of common modes unpacked");
  // desc.add<unsigned int>("flaggedECONDMax", 200)->setComment("maximum number of flagged ECON-Ds");
  // desc.add<unsigned int>("numERxsInECOND", 12)->setComment("number of eRxs in each ECON-D payload");
  // desc.add<edm::ESInputTag>("configSource", edm::ESInputTag(""))
  //     ->setComment("label for HGCalConfigESSourceFromYAML reader");
  // desc.add<edm::ESInputTag>("moduleInfoSource", edm::ESInputTag(""))->setComment("label for HGCalModuleInfoESSource");
  desc.add<bool>("fixCalibChannel", true)
      ->setComment("FIXME: always treat calib channels in characterization mode; to be fixed in ROCv3b");
  descriptions.add("hgcalDigis", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalRawToDigi);
