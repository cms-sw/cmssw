/** \class GEMDigiToRawModule
 *  \packer for gem
 *  \based on CSCDigiToRawModule
 *  \author J. Lee - UoS
 */

#include "CondFormats/DataRecord/interface/GEMChMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMChMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/GEMDigi/interface/GEMAMC13.h"
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

class GEMDigiToRawModule : public edm::global::EDProducer<edm::RunCache<GEMChMap>> {
public:
  /// Constructor
  GEMDigiToRawModule(const edm::ParameterSet& pset);

  // global::EDProducer
  std::shared_ptr<GEMChMap> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override;
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override{};

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  int event_type_;
  edm::EDGetTokenT<GEMDigiCollection> digi_token;
  edm::ESGetToken<GEMChMap, GEMChMapRcd> gemChMapToken_;
  bool useDBEMap_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMDigiToRawModule);

GEMDigiToRawModule::GEMDigiToRawModule(const edm::ParameterSet& pset)
    : event_type_(pset.getParameter<int>("eventType")),
      digi_token(consumes<GEMDigiCollection>(pset.getParameter<edm::InputTag>("gemDigi"))),
      useDBEMap_(pset.getParameter<bool>("useDBEMap")) {
  produces<FEDRawDataCollection>();
  if (useDBEMap_) {
    gemChMapToken_ = esConsumes<GEMChMap, GEMChMapRcd, edm::Transition::BeginRun>();
  }
}

void GEMDigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gemDigi", edm::InputTag("simMuonGEMDigis"));
  desc.add<int>("eventType", 0);
  desc.add<bool>("useDBEMap", false);
  descriptions.add("gemPackerDefault", desc);
}

std::shared_ptr<GEMChMap> GEMDigiToRawModule::globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const {
  if (useDBEMap_) {
    const auto& eMap = iSetup.getData(gemChMapToken_);
    auto gemChMap = std::make_shared<GEMChMap>(eMap);
    return gemChMap;
  } else {
    // no EMap in DB, using dummy
    auto gemChMap = std::make_shared<GEMChMap>();
    gemChMap->setDummy();
    return gemChMap;
  }
}

void GEMDigiToRawModule::produce(edm::StreamID iID, edm::Event& iEvent, edm::EventSetup const&) const {
  auto fedRawDataCol = std::make_unique<FEDRawDataCollection>();

  edm::Handle<GEMDigiCollection> gemDigis;
  iEvent.getByToken(digi_token, gemDigis);
  if (!gemDigis.isValid()) {
    iEvent.put(std::move(fedRawDataCol));
    return;
  }

  auto gemChMap = runCache(iEvent.getRun().index());

  std::vector<std::unique_ptr<GEMAMC13>> amc13s;
  amc13s.reserve(FEDNumbering::MAXGEMFEDID - FEDNumbering::MINGEMFEDID + 1);

  int LV1_id = iEvent.id().event();
  uint8_t BX_id(iEvent.bunchCrossing());
  int OrN = iEvent.orbitNumber();

  // making map of bx GEMDigiCollection
  // each bx will be saved as new GEMAMC13, so GEMDigiCollection needs to be split into bx
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

  for (unsigned int fedId = FEDNumbering::MINGEMFEDID; fedId <= FEDNumbering::MAXGEMFEDID; ++fedId) {
    uint32_t amc13EvtLength = 0;
    std::unique_ptr<GEMAMC13> amc13 = std::make_unique<GEMAMC13>();

    for (uint8_t amcNum = 0; amcNum <= GEMChMap::maxAMCs_; ++amcNum) {
      uint32_t amcSize = 0;
      std::unique_ptr<GEMAMC> amc = std::make_unique<GEMAMC>();

      for (uint8_t gebId = 0; gebId <= GEMChMap::maxGEBs_; ++gebId) {
        std::unique_ptr<GEMOptoHybrid> optoH = std::make_unique<GEMOptoHybrid>();

        if (!gemChMap->isValidChamber(fedId, amcNum, gebId))
          continue;

        auto geb_dc = gemChMap->chamberPos(fedId, amcNum, gebId);
        GEMDetId cid = geb_dc.detId;
        int chamberType = geb_dc.chamberType;

        auto vfats = gemChMap->getVfats(chamberType);

        for (auto vfatId : vfats) {
          auto iEtas = gemChMap->getIEtas(chamberType, vfatId);
          for (auto iEta : iEtas) {
            GEMDetId gemId(cid.region(), cid.ring(), cid.station(), cid.layer(), cid.chamber(), iEta);

            for (auto const& gemBx : gemBxMap) {
              int bc = BX_id + gemBx.first;

              bool hasDigi = false;
              uint64_t lsData = 0;  ///<channels from 1to64
              uint64_t msData = 0;  ///<channels from 65to128

              GEMDigiCollection inBxGemDigis = gemBx.second;
              const GEMDigiCollection::Range& range = inBxGemDigis.get(gemId);

              for (GEMDigiCollection::const_iterator digiIt = range.first; digiIt != range.second; ++digiIt) {
                const GEMDigi& digi = (*digiIt);

                int strip = digi.strip();

                hasDigi = true;

                if (!gemChMap->isValidStrip(chamberType, iEta, strip))
                  continue;
                auto chMap = gemChMap->getChannel(chamberType, iEta, strip);

                if (chMap.vfatAdd != vfatId)
                  continue;

                if (chMap.chNum < 64)
                  lsData |= 1UL << chMap.chNum;
                else
                  msData |= 1UL << (chMap.chNum - 64);

                LogDebug("GEMDigiToRawModule")
                    << "fed: " << fedId << " amc:" << int(amcNum) << " geb:" << int(gebId) << " vfat id:" << int(vfatId)
                    << ",type:" << chamberType << " id:" << gemId << " ch:" << chMap.chNum << " st:" << digi.strip()
                    << " bx:" << digi.bx();
              }

              if (!hasDigi)
                continue;
              // only make vfat with hits
              amcSize += 3;
              int vfatVersion = (chamberType < 10) ? 2 : 3;
              auto vfat = std::make_unique<GEMVFAT>(vfatVersion, bc, LV1_id, vfatId, lsData, msData);
              optoH->addVFAT(*vfat);
            }
          }
        }  // end of vfats in GEB

        if (!optoH->vFATs()->empty()) {
          amcSize += 2;
          optoH->setChamberHeader(optoH->vFATs()->size() * 3, gebId);
          optoH->setChamberTrailer(LV1_id, BX_id, optoH->vFATs()->size() * 3);
          amc->addGEB(*optoH);
        }
      }  // end of GEB loop

      if (!amc->gebs()->empty()) {
        amcSize += 5;
        amc->setAMCheader1(amcSize, BX_id, LV1_id, amcNum);
        amc->setAMCheader2(amcNum, OrN, 1);
        amc->setGEMeventHeader(amc->gebs()->size(), 0);
        amc13->addAMCpayload(*amc);
        // AMC header in GEMAMC13
        amc13->addAMCheader(amcSize, 0, amcNum, 0);
        amc13EvtLength += amcSize + 1;  // AMC data size + AMC header size
      }
    }  // end of AMC loop

    if (!amc13->getAMCpayloads()->empty()) {
      // CDFHeader
      amc13->setCDFHeader(event_type_, LV1_id, BX_id, fedId);
      // AMC13header
      uint8_t nAMC = amc13->getAMCpayloads()->size();
      amc13->setAMC13Header(1, nAMC, OrN);
      amc13->setAMC13Trailer(BX_id, LV1_id, BX_id);
      //CDF trailer
      uint32_t EvtLength = amc13EvtLength + 4;  // 2 header and 2 trailer
      amc13->setCDFTrailer(EvtLength);
      amc13s.emplace_back(std::move(amc13));
    }  // finished making amc13 data
  }    // end of FED loop

  // read out amc13s into fedRawData
  for (const auto& amc13e : amc13s) {
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
    LogDebug("GEMDigiToRawModule") << "fedId:" << amc13e->sourceId() << " words:" << words.size();
  }

  iEvent.put(std::move(fedRawDataCol));
}
