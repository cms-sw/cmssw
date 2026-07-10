#include "EventFilter/L1ScoutingRawToDigi/plugins/ScBMTFRawToDigi.h"

ScBMTFRawToDigi::ScBMTFRawToDigi(const edm::ParameterSet& iConfig) {
  srcInputTag_ = iConfig.getParameter<edm::InputTag>("srcInputTag");
  sourceIdList_ = iConfig.getParameter<std::vector<int>>("sourceIdList");

  // initialize orbit buffer for BX 1->3564;
  orbitBuffer_ = std::vector<std::vector<l1ScoutingRun3::BMTFStub>>(3565);
  for (auto& bxVec : orbitBuffer_) {
    bxVec.reserve(32);
  }
  nStubsOrbit_ = 0;

  produces<l1ScoutingRun3::BMTFStubOrbitCollection>("BMTFStub").setBranchAlias("BMTFStubOrbitCollection");
  rawToken_ = consumes<SDSRawDataCollection>(srcInputTag_);
}

ScBMTFRawToDigi::~ScBMTFRawToDigi() {}

void ScBMTFRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<SDSRawDataCollection> ScoutingRawDataCollection;
  iEvent.getByToken(rawToken_, ScoutingRawDataCollection);

  std::unique_ptr<l1ScoutingRun3::BMTFStubOrbitCollection> unpackedStubs(new l1ScoutingRun3::BMTFStubOrbitCollection);

  for (const auto& sdsId : sourceIdList_) {
    if ((sdsId < SDSNumbering::BmtfMinSDSID) || (sdsId > SDSNumbering::BmtfMaxSDSID))
      edm::LogError("ScBMTFRawToDigi::produce")
          << "Provided a source ID outside the expected range: " << sdsId << ", expected range ["
          << SDSNumbering::BmtfMinSDSID << ", " << SDSNumbering::BmtfMaxSDSID;
    const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(sdsId);
    size_t orbitSize = sourceRawData.size();

    if (sourceRawData.size() == 0) {
      LogDebug("L1Scout") << "No raw data for BMTF FED " << sdsId << "\n";
    }

    // unpack current orbit and store data into the orbitBufferr
    unpackOrbit(sourceRawData.data(), orbitSize, sdsId);
  }

  // fill orbit collection and clear the Bx buffer vector
  unpackedStubs->fillAndClear(orbitBuffer_, nStubsOrbit_);

  // store collection in the event
  iEvent.put(std::move(unpackedStubs), "BMTFStub");
}

void ScBMTFRawToDigi::unpackOrbit(const unsigned char* buf, size_t len, int sdsId) {
  // reset counters
  nStubsOrbit_ = 0;

  size_t pos = 0;

  while (pos < len) {
    assert(pos + 4 <= len);

    l1ScoutingRun3::bmtf::block* bl = (l1ScoutingRun3::bmtf::block*)(buf + pos);

    unsigned bx = bl->bx;
    unsigned orbit = (bl->orbit) & 0x7FFFFFFF;
    unsigned sCount = (bl->header) & 0xff;

    size_t pos_increment = 12 + sCount * 8;

    assert(pos_increment <= len);

    pos += 12;  // header

    LogDebug("L1Scout") << " BMTF #" << sdsId << " Orbit " << orbit << ", BX -> " << bx << ", nStubs -> " << sCount
                        << "\n";

    // Unpack stubs for the current pair (BX, sector)
    int32_t phi, phiB, tag, qual, eta, qeta, station, wheel, sector;

    // map for station and wheel, to find chambers with 2 stubs
    std::vector<std::vector<bool>> stwh_matrix(4, std::vector<bool>(5, false));
    for (unsigned int i = 0; i < sCount; i++) {
      uint64_t stub_raw = *(uint64_t*)(buf + pos);
      pos += 8;

      phi = ((stub_raw >> l1ScoutingRun3::bmtf::shiftsStubs::phi) & l1ScoutingRun3::bmtf::masksStubs::phi);
      phiB = ((stub_raw >> l1ScoutingRun3::bmtf::shiftsStubs::phiB) & l1ScoutingRun3::bmtf::masksStubs::phiB);
      qual = ((stub_raw >> l1ScoutingRun3::bmtf::shiftsStubs::qual) & l1ScoutingRun3::bmtf::masksStubs::qual);
      eta = ((stub_raw >> l1ScoutingRun3::bmtf::shiftsStubs::eta) & l1ScoutingRun3::bmtf::masksStubs::eta);
      qeta = ((stub_raw >> l1ScoutingRun3::bmtf::shiftsStubs::qeta) & l1ScoutingRun3::bmtf::masksStubs::qeta);
      station =
          ((stub_raw >> l1ScoutingRun3::bmtf::shiftsStubs::station) & l1ScoutingRun3::bmtf::masksStubs::station) + 1;
      wheel = ((stub_raw >> l1ScoutingRun3::bmtf::shiftsStubs::wheel) & l1ScoutingRun3::bmtf::masksStubs::wheel);
      sector = sdsId - SDSNumbering::BmtfMinSDSID;

      if (stwh_matrix[station - 1][wheel + 2] == false) {
        tag = 1;
      } else {
        tag = 0;
      }
      stwh_matrix[station - 1][wheel + 2] = true;

      phi = phi >= 2048 ? phi - 4096 : phi;
      phiB = phiB >= 512 ? phiB - 1024 : phiB;
      wheel = wheel >= 4 ? wheel - 8 : wheel;

      l1ScoutingRun3::BMTFStub stub(phi, phiB, qual, eta, qeta, station, wheel, sector, tag);
      orbitBuffer_[bx].push_back(stub);
      nStubsOrbit_++;

      LogDebug("L1Scout") << "Stub " << i << ", raw: 0x" << std::hex << stub_raw << std::dec << "\n"
                          << "\tPhi: " << phi << "\n"
                          << "\tPhiB: " << phiB << "\n"
                          << "\tQuality: " << qual << "\n"
                          << "\tEta: " << eta << "\n"
                          << "\tQEta: " << qeta << "\n"
                          << "\tStation: " << station << "\n"
                          << "\tWheel: " << wheel << "\n"
                          << "\tSector: " << sector << "\n"
                          << "\tTag: " << tag << "\n";
    }
  }  // end orbit while loop
}

void ScBMTFRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScBMTFRawToDigi);
