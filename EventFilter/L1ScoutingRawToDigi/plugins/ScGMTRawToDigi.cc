#include "EventFilter/L1ScoutingRawToDigi/plugins/ScGMTRawToDigi.h"

ScGMTRawToDigi::ScGMTRawToDigi(const edm::ParameterSet& iConfig) {
  srcInputTag = iConfig.getParameter<edm::InputTag>("srcInputTag");
  skipInterm_ = iConfig.getParameter<bool>("skipInterm");

  // initialize orbit buffer for BX 1->3564;
  orbitBuffer_ = std::vector<std::vector<l1ScoutingRun3::Muon>>(3565);
  for (auto& bxVec : orbitBuffer_) {
    bxVec.reserve(8);
  }
  nMuonsOrbit_ = 0;

  produces<l1ScoutingRun3::MuonOrbitCollection>("Muon").setBranchAlias("MuonOrbitCollection");
  rawToken = consumes<SDSRawDataCollection>(srcInputTag);
}

ScGMTRawToDigi::~ScGMTRawToDigi() {}

void ScGMTRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<SDSRawDataCollection> ScoutingRawDataCollection;
  iEvent.getByToken(rawToken, ScoutingRawDataCollection);

  const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(SDSNumbering::GmtSDSID);
  size_t orbitSize = sourceRawData.size();

  std::unique_ptr<l1ScoutingRun3::MuonOrbitCollection> unpackedMuons(new l1ScoutingRun3::MuonOrbitCollection);

  if (sourceRawData.size() == 0) {
    LogDebug("L1Scout") << "No raw data for GMT FED\n";
  }

  // unpack current orbit and store data into the orbitBufferr
  unpackOrbit(sourceRawData.data(), orbitSize);

  // fill orbit collection and clear the Bx buffer vector
  unpackedMuons->fillAndClear(orbitBuffer_, nMuonsOrbit_);

  // store collection in the event
  iEvent.put(std::move(unpackedMuons), "Muon");
}

void ScGMTRawToDigi::unpackOrbit(const unsigned char* buf, size_t len) {
  // reset counters
  nMuonsOrbit_ = 0;

  size_t pos = 0;

  while (pos < len) {
    assert(pos + 4 <= len);

    // get BX header
    uint32_t header = *((uint32_t*)(buf + pos));
    pos += 4;
    // count mA and mB
    uint32_t mAcount = (header & l1ScoutingRun3::header_masks::mAcount) >> l1ScoutingRun3::header_shifts::mAcount;
    uint32_t mBcount = (header & l1ScoutingRun3::header_masks::mBcount) >> l1ScoutingRun3::header_shifts::mBcount;

    // declare block to read
    l1ScoutingRun3::ugmt::block* bl = (l1ScoutingRun3::ugmt::block*)(buf + pos);
    pos += 4 + 4 + (mAcount + mBcount) * 12;
    assert(pos <= len);

    uint32_t orbit = bl->orbit & 0x7FFFFFFF;
    uint32_t bx = bl->bx;

    LogDebug("L1Scout") << "GMT Orbit " << orbit << ", BX -> " << bx << ", nMuons -> " << mAcount + mBcount;

    // Unpack muons for this BX
    orbitBuffer_[bx].reserve(mAcount + mBcount);

    for (unsigned int i = 0; i < mAcount + mBcount; i++) {
      uint32_t interm =
          (bl->mu[i].extra >> l1ScoutingRun3::ugmt::shiftsMuon::interm) & l1ScoutingRun3::ugmt::masksMuon::interm;
      if ((interm == 1) && (skipInterm_)) {
        LogDebug("L1Scout") << " -> Excluding intermediate muon\n";
        continue;
      }

      uint32_t index =
          (bl->mu[i].s >> l1ScoutingRun3::ugmt::shiftsMuon::index) & l1ScoutingRun3::ugmt::masksMuon::index;
      uint32_t ietaextu =
          (bl->mu[i].f >> l1ScoutingRun3::ugmt::shiftsMuon::etaext) & l1ScoutingRun3::ugmt::masksMuon::etaextv;
      int32_t ietaext;
      if (((bl->mu[i].f >> l1ScoutingRun3::ugmt::shiftsMuon::etaext) & l1ScoutingRun3::ugmt::masksMuon::etaexts) != 0) {
        ietaext = ietaextu -= 256;
      } else {
        ietaext = ietaextu;
      }

      // extract pt and quality and apply cut if required
      int32_t iptuncon =
          (bl->mu[i].s >> l1ScoutingRun3::ugmt::shiftsMuon::ptuncon) & l1ScoutingRun3::ugmt::masksMuon::ptuncon;
      int32_t ipt = (bl->mu[i].f >> l1ScoutingRun3::ugmt::shiftsMuon::pt) & l1ScoutingRun3::ugmt::masksMuon::pt;
      if ((ipt - 1) < 0) {
        continue;
      }
      uint32_t qual = (bl->mu[i].f >> l1ScoutingRun3::ugmt::shiftsMuon::qual) & l1ScoutingRun3::ugmt::masksMuon::qual;

      // extract integer value for extrapolated phi
      int32_t iphiext =
          ((bl->mu[i].f >> l1ScoutingRun3::ugmt::shiftsMuon::phiext) & l1ScoutingRun3::ugmt::masksMuon::phiext);

      // extract integer value for extrapolated phi
      int32_t idxy = ((bl->mu[i].s >> l1ScoutingRun3::ugmt::shiftsMuon::dxy) & l1ScoutingRun3::ugmt::masksMuon::dxy);

      // extract iso bits and charge
      uint32_t iso = (bl->mu[i].s >> l1ScoutingRun3::ugmt::shiftsMuon::iso) & l1ScoutingRun3::ugmt::masksMuon::iso;
      int32_t chrg = 0;
      if (((bl->mu[i].s >> l1ScoutingRun3::ugmt::shiftsMuon::chrgv) & l1ScoutingRun3::ugmt::masksMuon::chrgv) == 1)
        chrg = ((bl->mu[i].s >> l1ScoutingRun3::ugmt::shiftsMuon::chrg) & l1ScoutingRun3::ugmt::masksMuon::chrg) == 1
                   ? -1
                   : 1;

      // extract eta and phi at muon station
      int32_t iphi = (bl->mu[i].s >> l1ScoutingRun3::ugmt::shiftsMuon::phi) & l1ScoutingRun3::ugmt::masksMuon::phi;
      uint32_t ieta1 =
          (bl->mu[i].extra >> l1ScoutingRun3::ugmt::shiftsMuon::eta1) & l1ScoutingRun3::ugmt::masksMuon::eta;
      uint32_t ieta2 =
          (bl->mu[i].extra >> l1ScoutingRun3::ugmt::shiftsMuon::eta2) & l1ScoutingRun3::ugmt::masksMuon::eta;

      uint32_t ieta_u;
      int32_t ieta;
      // checking if raw eta should be taken from muon 1 or muon 2
      if ((bl->mu[i].extra & 0x1) == 0) {
        ieta_u = ieta1;
      } else {
        ieta_u = ieta2;
      }

      // two's complement
      if (ieta_u > 256) {
        ieta = ieta_u - 512;
      } else {
        ieta = ieta_u;
      }

      // increment muon counter
      nMuonsOrbit_++;

      l1ScoutingRun3::Muon muon(ipt, ieta, iphi, qual, chrg, chrg != 0, iso, index, ietaext, iphiext, iptuncon, idxy);

      orbitBuffer_[bx].push_back(muon);

      if (edm::MessageDrop::instance()->debugEnabled) {
        std::ostringstream os;
        LogDebug("L1Scout") << "--- Muon " << i << " ---\n"
                            << "  Raw f:     0x" << std::hex << bl->mu[i].f << std::dec << "\n"
                            << "  Raw s:     0x" << std::hex << bl->mu[i].s << std::dec << "\n"
                            << "  Raw extra: 0x" << std::hex << bl->mu[i].extra << std::dec << "\n";
        printMuon(muon, os);
        LogDebug("L1Scout") << os.str();
      }

    }  // end of bx

  }  // end orbit while loop
}

void ScGMTRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("srcInputTag", edm::InputTag("rawDataCollector"));
  desc.add<bool>("skipInterm", true);
  desc.addUntracked<bool>("debug", false);
  descriptions.add("ScGMTRawToDigi", desc);
}

DEFINE_FWK_MODULE(ScGMTRawToDigi);
