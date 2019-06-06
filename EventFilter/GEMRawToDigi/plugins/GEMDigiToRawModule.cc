/** \packer for gem
 *  \author J. Lee - UoS
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/GEMRawToDigi/plugins/GEMDigiToRawModule.h"

using namespace gem;

GEMDigiToRawModule::GEMDigiToRawModule(const edm::ParameterSet& pset)
    : event_type_(pset.getParameter<int>("eventType")),
      digi_token(consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("gemDigi"))),
      useDBEMap_(pset.getParameter<bool>("useDBEMap")) {
  produces<FEDRawDataCollection>();
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
    edm::ESHandle<GEMeMap> gemEMapRcd;
    iSetup.get<GEMeMapRcd>().get(gemEMapRcd);
    auto gemEMap = std::make_unique<GEMeMap>(*(gemEMapRcd.product()));
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

  for (unsigned int fedId = FEDNumbering::MINGEMFEDID; fedId <= FEDNumbering::MAXGEMFEDID; ++fedId) {
    std::unique_ptr<AMC13Event> amc13Event = std::make_unique<AMC13Event>();

    for (uint8_t amcNum = 0; amcNum < GEMeMap::maxAMCs_; ++amcNum) {
      std::unique_ptr<AMCdata> amcData = std::make_unique<AMCdata>();

      for (uint8_t gebId = 0; gebId < GEMeMap::maxGEBs_; ++gebId) {
        std::unique_ptr<GEBdata> gebData = std::make_unique<GEBdata>();
        GEMROMapping::chamEC geb_ec{fedId, amcNum, gebId};

        if (!gemROMap->isValidChamber(geb_ec))
          continue;
        GEMROMapping::chamDC geb_dc = gemROMap->chamberPos(geb_ec);

        auto vfats = gemROMap->getVfats(geb_dc.detId);
        for (auto vfat_ec : vfats) {
          GEMROMapping::vfatDC vfat_dc = gemROMap->vfatPos(vfat_ec);
          GEMDetId gemId = vfat_dc.detId;
          uint16_t vfatId = vfat_ec.vfatAdd;

          for (uint16_t bc = 0; bc < 2 * GEMeMap::amcBX_; ++bc) {
            bool hasDigi = false;

            uint64_t lsData = 0;  ///<channels from 1to64
            uint64_t msData = 0;  ///<channels from 65to128

            GEMDigiCollection::Range range = gemDigis->get(gemId);
            for (GEMDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
              const GEMDigi& digi = (*digiIt);
              if (digi.bx() != bc - GEMeMap::amcBX_)
                continue;

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
            auto vfatData = std::make_unique<VFATdata>(geb_dc.vfatVer, bc, 0, vfatId, lsData, msData);
            gebData->addVFAT(*vfatData);
          }

        }  // end of vfats in GEB

        if (!gebData->vFATs()->empty()) {
          gebData->setChamberHeader(gebData->vFATs()->size() * 3, gebId);
          gebData->setChamberTrailer(0, 0, gebData->vFATs()->size() * 3);
          amcData->addGEB(*gebData);
        }

      }  // end of GEB loop

      if (!amcData->gebs()->empty()) {
        amcData->setAMCheader1(0, GEMeMap::amcBX_, 0, amcNum);
        amcData->setAMCheader2(amcNum, 0, 1);
        amcData->setGEMeventHeader(amcData->gebs()->size(), 0);
        amc13Event->addAMCpayload(*amcData);
      }

    }  // end of AMC loop

    if (!amc13Event->getAMCpayloads()->empty()) {
      // CDFHeader
      uint32_t LV1_id = iEvent.id().event();
      uint16_t BX_id = iEvent.bunchCrossing();
      amc13Event->setCDFHeader(event_type_, LV1_id, BX_id, fedId);

      // AMC13header
      uint8_t CalTyp = 1;
      uint8_t nAMC = amc13Event->getAMCpayloads()->size();
      uint32_t OrN = 2;
      amc13Event->setAMC13Header(CalTyp, nAMC, OrN);

      for (unsigned short i = 0; i < amc13Event->nAMC(); ++i) {
        uint32_t AMC_size = 0;
        uint8_t Blk_No = 0;
        uint8_t AMC_No = 0;
        uint16_t BoardID = 0;
        amc13Event->addAMCheader(AMC_size, Blk_No, AMC_No, BoardID);
      }

      //AMC13 trailer
      uint8_t Blk_NoT = 0;
      uint8_t LV1_idT = 0;
      uint16_t BX_idT = BX_id;
      amc13Event->setAMC13Trailer(Blk_NoT, LV1_idT, BX_idT);
      //CDF trailer
      uint32_t EvtLength = 0;
      amc13Event->setCDFTrailer(EvtLength);
      amc13Events.emplace_back(std::move(amc13Event));
    }  // finished making amc13Event data

  }  // end of FED loop

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
    for (const auto& word : words)
      *(w++) = word;

    LogDebug("GEMDigiToRawModule") << " words " << words.size();
  }

  iEvent.put(std::move(fedRawDataCol));
}
