#include "EventFilter/L1ScoutingRawToDigi/plugins/ScBMTFRawToDigi.h"

ScBMTFRawToDigi::ScBMTFRawToDigi(const edm::ParameterSet& iConfig) {
  using namespace edm;
  srcInputTag = iConfig.getParameter<InputTag>("srcInputTag");
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);

  // initialize orbit buffer for BX 1->3564;
  orbitBuffer_ = std::vector<std::vector<l1ScoutingRun3::BMTFStub>>(3565);
  for (auto& bxVec : orbitBuffer_) {
    bxVec.reserve(32);
  }
  nStubsOrbit_ = 0;

  produces<l1ScoutingRun3::BMTFStubOrbitCollection>().setBranchAlias("BMTFStubOrbitCollection");
  rawToken = consumes<SDSRawDataCollection>(srcInputTag);
}

ScBMTFRawToDigi::~ScBMTFRawToDigi(){};

void ScBMTFRawToDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<SDSRawDataCollection> ScoutingRawDataCollection;
  iEvent.getByToken(rawToken, ScoutingRawDataCollection);

  std::unique_ptr<l1ScoutingRun3::BMTFStubOrbitCollection> unpackedStubs(new l1ScoutingRun3::BMTFStubOrbitCollection);

  for (unsigned int sdsId=SDSNumbering::BmtfMinSDSID; sdsId<SDSNumbering::BmtfMaxSDSID; sdsId++) {
    // get data and orbit size from i^th source
    const FEDRawData& sourceRawData = ScoutingRawDataCollection->FEDData(sdsId);
    size_t orbitSize = sourceRawData.size();

    if((sourceRawData.size()==0) && debug_){
      std::cout << "No raw data for BMTF FED " << sdsId << std::endl;
    }
    
    // unpack current orbit and store data into the orbitBufferr
    unpackOrbit(sourceRawData.data(), orbitSize, sdsId);
  }

  // fill orbit collection and clear the Bx buffer vector
  unpackedStubs->fillAndClear(orbitBuffer_, nStubsOrbit_);

  // store collection in the event
  iEvent.put(std::move(unpackedStubs));
}

void ScBMTFRawToDigi::unpackOrbit(const unsigned char* buf, size_t len, unsigned int sdsId) {
  using namespace l1ScoutingRun3;

  // reset counters
  nStubsOrbit_ = 0;

  size_t pos = 0;

  while (pos < len) {
    assert(pos + 4 <= len);

    bmtf::block *bl = (bmtf::block *)(buf + pos);
    
    unsigned bx = bl->bx;
    unsigned orbit = (bl->orbit) & 0x7FFFFFFF;
    unsigned sCount = (bl->header) & 0xff;

    pos += 12 + sCount*8;

    assert(pos <= len);

    if (debug_){
      std::cout  << " BMTF #" << sdsId << " Orbit " << orbit << ", BX -> "<< bx << ", nStubs -> " << sCount << std::endl;
      
    }

    // Unpack stubs for the current pair (BX, sector)
    int32_t phi, phiB, tag, qual, eta, qeta, station, wheel, sector;
    
    // map for station and wheel, to find chambers with 2 stubs
    std::vector<std::vector<bool>> stwh_matrix(4, std::vector<bool>(5,false));
    for (unsigned int i=0; i<sCount; i++) {

      phi      = ((bl->stub[i] >> bmtf::shiftsStubs::phi     ) & bmtf::masksStubs::phi     );
      phiB     = ((bl->stub[i] >> bmtf::shiftsStubs::phiB    ) & bmtf::masksStubs::phiB    );
      qual     = ((bl->stub[i] >> bmtf::shiftsStubs::qual    ) & bmtf::masksStubs::qual    );
      eta      = ((bl->stub[i] >> bmtf::shiftsStubs::eta     ) & bmtf::masksStubs::eta     );
      qeta     = ((bl->stub[i] >> bmtf::shiftsStubs::qeta    ) & bmtf::masksStubs::qeta    );
      station  = ((bl->stub[i] >> bmtf::shiftsStubs::station ) & bmtf::masksStubs::station ) + 1;
      wheel    = ((bl->stub[i] >> bmtf::shiftsStubs::wheel   ) & bmtf::masksStubs::wheel   );
      sector   = sdsId - SDSNumbering::BmtfMinSDSID;
      
      if (stwh_matrix[station-1][wheel+2]==false) {
        tag = 1;
      } else {
        tag = 0;
      }
      stwh_matrix[station-1][wheel+2] = true;

      phi      = phi   >= 2048 ? phi   - 4096 : phi;
      phiB     = phiB  >=  512 ? phiB  - 1024 : phiB;
      wheel    = wheel >=    4 ? wheel -    8 : wheel;

      BMTFStub stub(phi, phiB, qual, eta, qeta, station, wheel, sector, tag);
      orbitBuffer_[bx].push_back(stub);
      nStubsOrbit_ ++;
    }
    

  }  // end orbit while loop
}

void ScBMTFRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ScBMTFRawToDigi);
