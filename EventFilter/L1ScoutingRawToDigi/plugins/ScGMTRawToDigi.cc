#include "EventFilter/L1ScoutingRawToDigi/plugins/ScGMTRawToDigi.h"

ScGMTRawToDigi::ScGMTRawToDigi(const edm::ParameterSet& iConfig) {
  using namespace edm;
  srcInputTag = iConfig.getParameter<InputTag>("srcInputTag");
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);

  // initialize orbit buffer for BX 1->3564;
  orbitBuffer_ = std::vector<std::vector<l1ScoutingRun3::Muon>>(3565);
  for (auto& bxVec : orbitBuffer_) {
    bxVec.reserve(8);
  }
  nMuonsOrbit_ = 0;

  produces<l1ScoutingRun3::MuonOrbitCollection>().setBranchAlias("MuonOrbitCollection");
  rawToken = consumes<SDSRawDataCollection>(srcInputTag);
}

ScGMTRawToDigi::~ScGMTRawToDigi(){};

void ScGMTRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<SDSRawDataCollection> ScoutingRawDataCollection;
  iEvent.getByToken(rawToken, ScoutingRawDataCollection);

  const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(SDSNumbering::GmtSDSID);
  size_t orbitSize = sourceRawData.size();

  std::unique_ptr<l1ScoutingRun3::MuonOrbitCollection> unpackedMuons(new l1ScoutingRun3::MuonOrbitCollection);

  if ((sourceRawData.size() == 0) && debug_) {
    std::cout << "No raw data for GMT FED\n";
  }

  // unpack current orbit and store data into the orbitBufferr
  unpackOrbit(sourceRawData.data(), orbitSize);

  // fill orbit collection and clear the Bx buffer vector
  unpackedMuons->fillAndClear(orbitBuffer_, nMuonsOrbit_);

  // store collection in the event
  iEvent.put(std::move(unpackedMuons));
}

void ScGMTRawToDigi::unpackOrbit(const unsigned char* buf, size_t len) {
  using namespace l1ScoutingRun3;

  // reset counters
  nMuonsOrbit_ = 0;

  size_t pos = 0;

  while (pos < len) {
    assert(pos + 4 <= len);

    // get BX header
    uint32_t header = *((uint32_t*)(buf + pos));
    pos += 4;
    // count mA and mB
    uint32_t mAcount = (header & header_masks::mAcount) >> header_shifts::mAcount;
    uint32_t mBcount = (header & header_masks::mBcount) >> header_shifts::mBcount;

    // declare block to read
    ugmt::block* bl = (ugmt::block*)(buf + pos);
    pos += 4 + 4 + (mAcount + mBcount) * 12;
    assert(pos <= len);

    uint32_t orbit = bl->orbit & 0x7FFFFFFF;
    uint32_t bx = bl->bx;

    if (debug_) {
      std::cout << "GMT Orbit " << orbit << ", BX -> " << bx << ", nMuons -> " << mAcount + mBcount << std::endl;
    }

    // Unpack muons for this BX
    orbitBuffer_[bx].reserve(mAcount + mBcount);

    for (unsigned int i = 0; i < mAcount + mBcount; i++) {
      uint32_t interm = (bl->mu[i].extra >> ugmt::shiftsMuon::interm) & ugmt::masksMuon::interm;
      if (interm == 1) {
        if (debug_) {
          std::cout << " -> Excluding intermediate muon\n";
        }
        continue;
      }

      uint32_t index = (bl->mu[i].s >> ugmt::shiftsMuon::index) & ugmt::masksMuon::index;
      uint32_t ietaextu = (bl->mu[i].f >> ugmt::shiftsMuon::etaext) & ugmt::masksMuon::etaextv;
      int32_t ietaext;
      if (((bl->mu[i].f >> ugmt::shiftsMuon::etaext) & ugmt::masksMuon::etaexts) != 0) {
        ietaext = ietaextu -= 256;
      } else {
        ietaext = ietaextu;
      }

      // extract pt and quality and apply cut if required
      int32_t iptuncon = (bl->mu[i].s >> ugmt::shiftsMuon::ptuncon) & ugmt::masksMuon::ptuncon;
      int32_t ipt = (bl->mu[i].f >> ugmt::shiftsMuon::pt) & ugmt::masksMuon::pt;
      if ((ipt - 1) < 0) {
        continue;
      }
      uint32_t qual = (bl->mu[i].f >> ugmt::shiftsMuon::qual) & ugmt::masksMuon::qual;
      if (qual == 0) {
        continue;
      }

      // extract integer value for extrapolated phi
      int32_t iphiext = ((bl->mu[i].f >> ugmt::shiftsMuon::phiext) & ugmt::masksMuon::phiext);

      // extract integer value for extrapolated phi
      int32_t idxy = ((bl->mu[i].s >> ugmt::shiftsMuon::dxy) & ugmt::masksMuon::dxy);

      // extract iso bits and charge
      uint32_t iso = (bl->mu[i].s >> ugmt::shiftsMuon::iso) & ugmt::masksMuon::iso;
      int32_t chrg = 0;
      if (((bl->mu[i].s >> ugmt::shiftsMuon::chrgv) & ugmt::masksMuon::chrgv) == 1)
        chrg = ((bl->mu[i].s >> ugmt::shiftsMuon::chrg) & ugmt::masksMuon::chrg) == 1 ? -1 : 1;

      // extract eta and phi at muon station
      int32_t iphi = (bl->mu[i].s >> ugmt::shiftsMuon::phi) & ugmt::masksMuon::phi;
      uint32_t ieta1 = (bl->mu[i].extra >> ugmt::shiftsMuon::eta1) & ugmt::masksMuon::eta;
      uint32_t ieta2 = (bl->mu[i].extra >> ugmt::shiftsMuon::eta2) & ugmt::masksMuon::eta;

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

      if (debug_) {
        std::cout << "--- Muon " << i << " ---\n";
        std::cout << "  Raw f:     0x" << std::hex << bl->mu[i].f << std::dec << "\n";
        std::cout << "  Raw s:     0x" << std::hex << bl->mu[i].s << std::dec << "\n";
        std::cout << "  Raw extra: 0x" << std::hex << bl->mu[i].extra << std::dec << "\n";
        printMuon(muon);
      }

    }  // end of bx

  }  // end orbit while loop

  //muons->flatten();
}

void ScGMTRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScGMTRawToDigi);
