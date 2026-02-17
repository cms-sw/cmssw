#include <ios>
#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/L1Scouting/interface/L1ScoutingCaloTower.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSNumbering.h"
#include "DataFormats/L1ScoutingRawData/interface/SDSRawDataCollection.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/blocks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/masks.h"
#include "EventFilter/L1ScoutingRawToDigi/interface/shifts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class ScCaloTowerRawToDigi : public edm::stream::EDProducer<> {
public:
  explicit ScCaloTowerRawToDigi(const edm::ParameterSet&);
  ~ScCaloTowerRawToDigi() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // BX per orbit
  static constexpr unsigned int NBX = 3564;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void unpackOrbit(const unsigned char* buf, size_t len, int sdsId);

  edm::EDGetTokenT<SDSRawDataCollection> const rawToken_;
  std::vector<int> const sourceIdList_;
  bool const debug_;

  // vector holding data for every bunch crossing
  // before filling the orbit collection
  std::vector<std::vector<l1ScoutingRun3::CaloTower>> orbitBuffer_;

  int nCaloTowersOrbit_;
};

ScCaloTowerRawToDigi::ScCaloTowerRawToDigi(const edm::ParameterSet& iConfig)
    : rawToken_(consumes(iConfig.getParameter<edm::InputTag>("srcInputTag"))),
      sourceIdList_(iConfig.getParameter<std::vector<int>>("sourceIdList")),
      debug_(iConfig.getUntrackedParameter<bool>("debug")),
      orbitBuffer_(NBX + 1),
      nCaloTowersOrbit_(0) {
  for (auto& bxVec : orbitBuffer_) {
    // reasonable upper estimate
    bxVec.reserve(4096);
  }

  for (auto const& sdsId : sourceIdList_) {
    if (sdsId < SDSNumbering::CaloTowerMinSDSID or sdsId > SDSNumbering::CaloTowerMaxSDSID) {
      edm::LogError("ScCaloTowerRawToDigi")
          << "Provided a source ID outside the expected range: " << sdsId << ", expected range ["
          << SDSNumbering::CaloTowerMinSDSID << ", " << SDSNumbering::CaloTowerMaxSDSID << "]";
    }
  }

  produces<l1ScoutingRun3::CaloTowerOrbitCollection>("CaloTower").setBranchAlias("CaloTowerOrbitCollection");
}

void ScCaloTowerRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup&) {
  auto const& scoutingRawDataCollection = iEvent.get(rawToken_);

  auto unpackedCaloTowers = std::make_unique<l1ScoutingRun3::CaloTowerOrbitCollection>();

  nCaloTowersOrbit_ = 0;

  for (const auto& sdsId : sourceIdList_) {
    const FEDRawData& sourceRawData = scoutingRawDataCollection.FEDData(sdsId);
    size_t orbitSize = sourceRawData.size();

    if (debug_ && (orbitSize == 0)) {
      edm::LogWarning("ScCaloTowerRawToDigi") << "No raw data for CaloTower FED " << sdsId;
    }

    // unpack current orbit and store data into the orbitBufferr
    unpackOrbit(sourceRawData.data(), orbitSize, sdsId);
  }

  // fill orbit collection and clear the Bx buffer vector
  unpackedCaloTowers->fillAndClear(orbitBuffer_, nCaloTowersOrbit_);

  // store collection in the event
  iEvent.put(std::move(unpackedCaloTowers), "CaloTower");
}

void ScCaloTowerRawToDigi::unpackOrbit(const unsigned char* buf, size_t len, int sdsId) {
  size_t pos = 0;
  const size_t blockHeaderSize = 3 * sizeof(uint32_t);

  while (pos < len) {
    if (pos + blockHeaderSize > len) {
      edm::LogError("ScCaloTowerRawToDigi") << "Corrupt data in sourceId " << sdsId << ", incomplete header";
      break;  // no sense trying to unpack further
    }
    const l1ScoutingRun3::calol1::block* bl = reinterpret_cast<const l1ScoutingRun3::calol1::block*>(buf + pos);

    unsigned int bx = bl->bx;
    unsigned int orbit = (bl->orbit) & 0x7FFFFFFF;
    unsigned int ctCount = bl->header;

    pos += blockHeaderSize;

    if ((pos + 4 * ctCount) > len) {
      edm::LogError("ScCaloTowerRawToDigi")
          << "Corrupt data in sourceId " << sdsId << ", orbit " << orbit << ", BX " << bx << ": expecting " << ctCount
          << " towers but only " << (len - pos) << " bytes left in the block.";
      break;  // no sense trying to unpack further
    }

    if (bx > NBX) {  // need this check as otherwise the code will crash later accessing orbitBuffer_
      edm::LogError("ScCaloTowerRawToDigi")
          << "Corrupt data in sourceId " << sdsId << ", orbit " << orbit << ", invalid BX " << bx;
      break;  // we could go to the next block, but if the data is corrupted it probably doesn't help
    }

    if (debug_) {
      edm::LogPrint("ScCaloTowerRawToDigi")
          << " CaloTower #" << sdsId << " Orbit " << orbit << ", BX -> " << bx << ", nCaloTowers -> " << ctCount;
    }

    // Unpack calo towers
    auto& bufferThisBX = orbitBuffer_[bx];
    int16_t ET, erBits, miscBits, eta, phi;

    const uint32_t* towerPtr = reinterpret_cast<const uint32_t*>(buf + pos);

    for (unsigned int i = 0; i < ctCount; ++i, ++towerPtr) {
      uint32_t ct_raw = *towerPtr;

      ET = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::ET) & l1ScoutingRun3::calol1::masksCaloTowers::ET);

      erBits = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::erBits) &
                l1ScoutingRun3::calol1::masksCaloTowers::erBits);

      miscBits = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::miscBits) &
                  l1ScoutingRun3::calol1::masksCaloTowers::miscBits);

      phi = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::phi) & l1ScoutingRun3::calol1::masksCaloTowers::phi);

      eta = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::eta) & l1ScoutingRun3::calol1::masksCaloTowers::eta);
      eta = eta >= 128 ? eta - 256 : eta;

      bufferThisBX.emplace_back(ET, erBits, miscBits, eta, phi);

      if (debug_) {
        edm::LogPrint("ScCaloTowerRawToDigi")
            << "Calo Tower " << i << ", raw: 0x" << std::hex << ct_raw << std::dec << "\n\tET: " << ET
            << "\n\tER bits: " << erBits << "\n\tMisc bits: " << miscBits << "\n\tEta: " << eta << "\n\tPhi: " << phi;
      }
    }

    pos += 4 * ctCount;
    nCaloTowersOrbit_ += ctCount;
  }  // end orbit while loop
}

void ScCaloTowerRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("srcInputTag", edm::InputTag("rawDataCollector"));

  std::vector<int> sourceIds;
  for (int id = SDSNumbering::CaloTowerMinSDSID; id <= SDSNumbering::CaloTowerMaxSDSID; ++id) {
    sourceIds.emplace_back(id);
  }
  desc.add<std::vector<int>>("sourceIdList", sourceIds);

  desc.addUntracked<bool>("debug", false);

  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ScCaloTowerRawToDigi);
