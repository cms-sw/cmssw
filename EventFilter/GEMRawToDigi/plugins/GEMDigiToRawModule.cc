/** \class GEMDigiToRawModule
 *  \packer for gem
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "CondFormats/DataRecord/interface/GEMeMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "CondFormats/GEMObjects/interface/GEMROMapping.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/GEMDigi/interface/AMC13Event.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class GEMDigiToRawModule : public edm::global::EDProducer<edm::RunCache<GEMROMapping>> {
public:
  /// Constructor
  GEMDigiToRawModule(const edm::ParameterSet& pset);

  // global::EDProducer
  std::shared_ptr<GEMROMapping> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override{};

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  int event_type_;
  edm::EDGetTokenT<GEMDigiCollection> digi_token;
  edm::ESGetToken<GEMeMap, GEMeMapRcd> gemEMapToken_;
  bool useDBEMap_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMDigiToRawModule);

using namespace gem;

GEMDigiToRawModule::GEMDigiToRawModule(const edm::ParameterSet& pset)
    : event_type_(pset.getParameter<int>("eventType")),
      digi_token(consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("gemDigi"))),
      useDBEMap_(pset.getParameter<bool>("useDBEMap")) {
  produces<FEDRawDataCollection>();
  if (useDBEMap_) {
    gemEMapToken_ = esConsumes<GEMeMap, GEMeMapRcd, edm::Transition::BeginRun>();
  }
}

void GEMDigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gemDigi", edm::InputTag("simMuonGEMDigis"));
  desc.add<int>("eventType", 0);
  desc.add<bool>("useDBEMap", false);
  descriptions.add("gemPackerDefault", desc);
}

std::shared_ptr<GEMROMapping> GEMDigiToRawModule::globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const {
  auto gemROmap = std::make_shared<GEMROMapping>();
  if (useDBEMap_) {
    const auto& eMap = iSetup.getData(gemEMapToken_);
    auto gemEMap = std::make_unique<GEMeMap>(eMap);
    gemEMap->convert(*gemROmap);
    gemEMap.reset();
  } else {
    // no EMap in DB, using dummy
    auto gemEMap = std::make_unique<GEMeMap>();
    gemEMap->convertDummy(*gemROmap);
    gemEMap.reset();
  }
  return gemROmap;
}

void GEMDigiToRawModule::produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto fedRawDataCol = std::make_unique<FEDRawDataCollection>();

  edm::Handle<GEMDigiCollection> gemDigis;
  iEvent.getByToken(digi_token, gemDigis);
  if (!gemDigis.isValid()) {
    iEvent.put(std::move(fedRawDataCol));
    return;
  }

  auto gemROMap = runCache(iEvent.getRun().index());

  std::vector<std::unique_ptr<AMC13Event>> amc13Events;
  amc13Events.reserve(FEDNumbering::MAXGEMFEDID - FEDNumbering::MINGEMFEDID + 1);

  int LV1_id = iEvent.id().event();
  uint8_t BX_id(iEvent.bunchCrossing());
  int OrN = iEvent.orbitNumber();

  // making map of bx GEMDigiCollection
  // each bx will be saved as new AMC13Event, so GEMDigiCollection needs to be split into bx
  std::map<int, GEMDigiCollection> gemBxMap;
  for (auto const& etaPart : *gemDigis) {
    GEMDetId gemId = etaPart.first;
    const GEMDigiCollection::Range& digis = etaPart.second;
    for (auto digi = digis.first; digi != digis.second; ++digi) {
      int bx = digi->bx();
      auto search = gemBxMap.find(bx);
      if (search != gemBxMap.end()) {
        search->second.insertDigi(gemId, *digi);
      } else {
        GEMDigiCollection newGDC;
        newGDC.insertDigi(gemId, *digi);
        gemBxMap.insert(std::pair<int, GEMDigiCollection>(bx, newGDC));
      }
    }
  }

  for (unsigned int fedId = FEDNumbering::MINGEMFEDID; fedId <= FEDNumbering::MAXME0FEDID; ++fedId) {
    uint32_t amc13EvtLength = 0;
    std::unique_ptr<AMC13Event> amc13Event = std::make_unique<AMC13Event>();

    for (uint8_t amcNum = 0; amcNum < GEMeMap::maxAMCs_; ++amcNum) {
      uint32_t amcSize = 0;
      std::unique_ptr<AMCdata> amcData = std::make_unique<AMCdata>();

      for (uint8_t gebId = 0; gebId < GEMeMap::maxGEBs_; ++gebId) {
        std::unique_ptr<GEBdata> gebData = std::make_unique<GEBdata>();
        GEMROMapping::chamEC geb_ec{fedId, amcNum, gebId};

        if (!gemROMap->isValidChamber(geb_ec))
          continue;
        GEMROMapping::chamDC geb_dc = gemROMap->chamberPos(geb_ec);

        auto vfats = gemROMap->getVfats(geb_dc.detId);
        for (auto const& vfat_ec : vfats) {
          GEMROMapping::vfatDC vfat_dc = gemROMap->vfatPos(vfat_ec);
          GEMDetId gemId = vfat_dc.detId;
          uint16_t vfatId = vfat_ec.vfatAdd;

          for (auto const& gemBx : gemBxMap) {
            int bc = BX_id + gemBx.first;

            bool hasDigi = false;
            uint64_t lsData = 0;  ///<channels from 1to64
            uint64_t msData = 0;  ///<channels from 65to128

            GEMDigiCollection inBxGemDigis = gemBx.second;
            const GEMDigiCollection::Range& range = inBxGemDigis.get(gemId);
            for (GEMDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
              const GEMDigi& digi = (*digiIt);

              int localStrip = digi.strip() - vfat_dc.localPhi * GEMeMap::maxChan_;

              // skip strips not in current vFat
              if (localStrip < 0 || localStrip > GEMeMap::maxChan_ - 1)
                continue;

              hasDigi = true;
              GEMROMapping::stripNum stMap = {vfat_dc.vfatType, localStrip};
              GEMROMapping::channelNum chMap = gemROMap->hitPos(stMap);

              if (chMap.chNum < 64)
                lsData |= 1UL << chMap.chNum;
              else
                msData |= 1UL << (chMap.chNum - 64);

              LogDebug("GEMDigiToRawModule")
                  << " fed: " << fedId << " amc:" << int(amcNum) << " geb:" << int(gebId)
                  << " vfat:" << vfat_dc.localPhi << ",type: " << vfat_dc.vfatType << " id:" << gemId
                  << " ch:" << chMap.chNum << " st:" << digi.strip() << " bx:" << digi.bx();
            }

            if (!hasDigi)
              continue;
            // only make vfat with hits
            amcSize += 3;
            auto vfatData = std::make_unique<VFATdata>(geb_dc.vfatVer, bc, 0, vfatId, lsData, msData);
            gebData->addVFAT(*vfatData);
          }

        }  // end of vfats in GEB

        if (!gebData->vFATs()->empty()) {
          amcSize += 2;
          gebData->setChamberHeader(gebData->vFATs()->size() * 3, gebId);
          gebData->setChamberTrailer(LV1_id, BX_id, gebData->vFATs()->size() * 3);
          amcData->addGEB(*gebData);
        }
      }  // end of GEB loop

      amcSize += 5;
      amcData->setAMCheader1(amcSize, BX_id, LV1_id, amcNum);
      amcData->setAMCheader2(amcNum, OrN, 1);
      amcData->setGEMeventHeader(amcData->gebs()->size(), 0);
      amc13Event->addAMCpayload(*amcData);
      // AMC header in AMC13Event
      amc13Event->addAMCheader(amcSize, 0, amcNum, 0);
      amc13EvtLength += amcSize + 1;  // AMC data size + AMC header size

    }  // end of AMC loop

    if (!amc13Event->getAMCpayloads()->empty()) {
      // CDFHeader
      amc13Event->setCDFHeader(event_type_, LV1_id, BX_id, fedId);
      // AMC13header
      uint8_t nAMC = amc13Event->getAMCpayloads()->size();
      amc13Event->setAMC13Header(1, nAMC, OrN);
      amc13Event->setAMC13Trailer(BX_id, LV1_id, BX_id);
      //CDF trailer
      uint32_t EvtLength = amc13EvtLength + 4;  // 2 header and 2 trailer
      amc13Event->setCDFTrailer(EvtLength);

      amc13Events.emplace_back(std::move(amc13Event));
    }  // finished making amc13Event data
  }    // end of FED loop

  // read out amc13Events into fedRawData
  for (const auto& amc13e : amc13Events) {
    std::vector<uint64_t> words;
    words.emplace_back(amc13e->getCDFHeader());
    words.emplace_back(amc13e->getAMC13Header());

    for (const auto& w : *amc13e->getAMCheaders())
      words.emplace_back(w);

    for (const auto& amc : *amc13e->getAMCpayloads()) {
      words.emplace_back(amc.getAMCheader1());
      words.emplace_back(amc.getAMCheader2());
      words.emplace_back(amc.getGEMeventHeader());

      for (const auto& geb : *amc.gebs()) {
        words.emplace_back(geb.getChamberHeader());

        for (const auto& vfat : *geb.vFATs()) {
          words.emplace_back(vfat.get_fw());
          words.emplace_back(vfat.get_sw());
          words.emplace_back(vfat.get_tw());
        }

        words.emplace_back(geb.getChamberTrailer());
      }

      words.emplace_back(amc.getGEMeventTrailer());
      words.emplace_back(amc.getAMCTrailer());
    }

    words.emplace_back(amc13e->getAMC13Trailer());
    words.emplace_back(amc13e->getCDFTrailer());

    FEDRawData& fedRawData = fedRawDataCol->FEDData(amc13e->sourceId());

    int dataSize = (words.size()) * sizeof(uint64_t);
    fedRawData.resize(dataSize);

    uint64_t* w = reinterpret_cast<uint64_t*>(fedRawData.data());
    for (const auto& word : words) {
      *(w++) = word;
    }
    LogDebug("GEMDigiToRawModule") << " words " << words.size();
  }

  iEvent.put(std::move(fedRawDataCol));
}
