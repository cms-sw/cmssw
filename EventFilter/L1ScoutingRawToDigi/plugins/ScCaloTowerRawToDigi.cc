#include "EventFilter/L1ScoutingRawToDigi/plugins/ScCaloTowerRawToDigi.h"

ScCaloTowerRawToDigi::ScCaloTowerRawToDigi(const edm::ParameterSet& iConfig) {
  srcInputTag_ = iConfig.getParameter<edm::InputTag>("srcInputTag");
  sourceIdList_ = iConfig.getParameter<std::vector<int>>("sourceIdList");

  // initialize orbit buffer for BX 1->3564;
  orbitBuffer_ = std::vector<std::vector<l1ScoutingRun3::CaloTower>>(3565);
  for (auto& bxVec : orbitBuffer_) {
    bxVec.reserve(4096);
  }
  nCaloTowersOrbit_ = 0;

  produces<l1ScoutingRun3::CaloTowerOrbitCollection>("CaloTower").setBranchAlias("CaloTowerOrbitCollection");
  rawToken_ = consumes<SDSRawDataCollection>(srcInputTag_);
}

ScCaloTowerRawToDigi::~ScCaloTowerRawToDigi() {}

void ScCaloTowerRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<SDSRawDataCollection> ScoutingRawDataCollection;
  iEvent.getByToken(rawToken_, ScoutingRawDataCollection);

  std::unique_ptr<l1ScoutingRun3::CaloTowerOrbitCollection> unpackedCaloTowers(
      new l1ScoutingRun3::CaloTowerOrbitCollection);

  for (const auto& sdsId : sourceIdList_) {
    if ((sdsId < SDSNumbering::CaloTowerMinSDSID) || (sdsId > SDSNumbering::CaloTowerMaxSDSID))
      edm::LogError("ScCaloTowerRawToDigi::produce")
          << "Provided a source ID outside the expected range: " << sdsId << ", expected range ["
          << SDSNumbering::CaloTowerMinSDSID << ", " << SDSNumbering::CaloTowerMaxSDSID;
    const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(sdsId);
    size_t orbitSize = sourceRawData.size();

    if (sourceRawData.size() == 0) {
      LogDebug("L1Scout") << "No raw data for CaloTower FED " << sdsId;
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
  // reset counters
  nCaloTowersOrbit_ = 0;

  size_t pos = 0;

  while (pos < len) {
    assert(pos + 4 <= len);

    l1ScoutingRun3::calol1::block* bl = (l1ScoutingRun3::calol1::block*)(buf + pos);

    unsigned bx = bl->bx;
    unsigned orbit = (bl->orbit) & 0x7FFFFFFF;
    unsigned ctCount = bl->header;

    size_t pos_increment = 12 + ctCount * 4;

    assert(pos_increment <= len);

    pos += 12;  // header

    LogDebug("L1Scout") << " CaloTower #" << sdsId << " Orbit " << orbit << ", BX -> " << bx << ", nCaloTowers -> "
                        << ctCount;

    // Unpack calo towers
    int16_t ET, erBits, miscBits, eta, phi;

    for (unsigned int i = 0; i < ctCount; i++) {
      uint64_t ct_raw = *(uint32_t*)(buf + pos);
      pos += 4;

      ET = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::ET) & l1ScoutingRun3::calol1::masksCaloTowers::ET);
      erBits = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::erBits) &
                l1ScoutingRun3::calol1::masksCaloTowers::erBits);
      miscBits = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::miscBits) &
                  l1ScoutingRun3::calol1::masksCaloTowers::miscBits);
      phi = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::phi) & l1ScoutingRun3::calol1::masksCaloTowers::phi);

      eta = ((ct_raw >> l1ScoutingRun3::calol1::shiftsCaloTowers::eta) & l1ScoutingRun3::calol1::masksCaloTowers::eta);
      eta = eta >= 128 ? eta - 256 : eta;

      l1ScoutingRun3::CaloTower ct(ET, erBits, miscBits, eta, phi);
      orbitBuffer_[bx].push_back(ct);
      nCaloTowersOrbit_++;

      LogDebug("L1Scout") << "Calo Tower " << i << ", raw: 0x" << std::hex << ct_raw << std::dec << "\tET: " << ET
                          << "\tER bits: " << erBits << "\tMisc bits: " << miscBits << "\tEta: " << eta
                          << "\tPhi: " << phi;
    }

  }  // end orbit while loop
}

void ScCaloTowerRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScCaloTowerRawToDigi);
