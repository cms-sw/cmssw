#include "CSCTFSingleGen.h"

//Framework stuff
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Digi
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

//Digi collections
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"

//Unique key
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include "CondFormats/CSCObjects/interface/CSCTriggerMappingFromFile.h"
#include <sstream>

CSCTFSingleGen::CSCTFSingleGen(const edm::ParameterSet& pset) : edm::EDProducer(), mapping(nullptr) {
  LogDebug("CSCTFSingleGen|ctor") << "Started ...";

  // Edges of the time window, which LCTs are put into (unlike tracks, which are always centred around 0):
  m_minBX = 3;
  m_maxBX = 9;

  endcap = pset.getUntrackedParameter<int>("endcap", 1);
  sector = pset.getUntrackedParameter<int>("sector", 1);
  subSector = pset.getUntrackedParameter<int>("subSector", 0);
  station = pset.getUntrackedParameter<int>("station", 1);
  cscId = pset.getUntrackedParameter<int>("cscId", 5);
  strip = 0;  //pset.getUntrackedParameter<int>("strip", -1);
  wireGroup = pset.getUntrackedParameter<int>("wireGroup", 48);
  pattern = pset.getUntrackedParameter<int>("pattern", 4);

  // As we use standard CSC digi containers, we have to initialize mapping:
  std::string mappingFile = pset.getUntrackedParameter<std::string>("mappingFile", "");
  if (mappingFile.length()) {
    LogDebug("CSCTFSingleGen|ctor") << "Define ``mapping'' only if you want to screw up real geometry";
    mapping = new CSCTriggerMappingFromFile(mappingFile);
  } else {
    LogDebug("CSCTFSingleGen|ctor") << "Generating default hw<->geometry mapping";
    class M : public CSCTriggerSimpleMapping {
      void fill(void) override {}
    };
    mapping = new M();
    for (int endcap = 1; endcap <= 2; endcap++)
      for (int station = 1; station <= 4; station++)
        for (int sector = 1; sector <= 6; sector++)
          for (int csc = 1; csc <= 9; csc++) {
            if (station == 1) {
              mapping->addRecord(endcap, station, sector, 1, csc, endcap, station, sector, 1, csc);
              mapping->addRecord(endcap, station, sector, 2, csc, endcap, station, sector, 2, csc);
            } else
              mapping->addRecord(endcap, station, sector, 0, csc, endcap, station, sector, 0, csc);
          }
  }

  strip = 0;

  produces<CSCCorrelatedLCTDigiCollection>();
  //  produces<CSCTriggerContainer<csctf::TrackStub> >("DT");
}

CSCTFSingleGen::~CSCTFSingleGen() {
  if (mapping)
    delete mapping;
}

void CSCTFSingleGen::produce(edm::Event& e, const edm::EventSetup& c) {
  // create the collection of CSC wire and strip digis as well as of DT stubs, which we receive from DTTF
  auto LCTProduct = std::make_unique<CSCCorrelatedLCTDigiCollection>();
  //  auto dtProduct = std::make_unique<CSCTriggerContainer<csctf::TrackStub>>();

  for (unsigned int tbin = 6; tbin < 7; tbin++) {
    //for(unsigned int FPGA=0; FPGA<5; FPGA++)
    for (unsigned int FPGA = 0; FPGA < 1; FPGA++)
      //for(unsigned int MPClink=1; MPClink<4; ++MPClink){
      for (unsigned int MPClink = 1; MPClink < 2; ++MPClink) {
        try {
          CSCDetId id = mapping->detId(endcap, station, sector, subSector, cscId, 0);
          // corrlcts now have no layer associated with them
          LCTProduct->insertDigi(id,
                                 CSCCorrelatedLCTDigi(0,
                                                      1,
                                                      15,
                                                      wireGroup,
                                                      strip,
                                                      pattern,
                                                      1,  //l_r
                                                      tbin,
                                                      MPClink,
                                                      0,
                                                      0,
                                                      cscId));
        } catch (cms::Exception& e) {
          edm::LogInfo("CSCTFSingleGen|produce")
              << e.what() << "Not adding digi to collection in event "
              << " (endcap=" << endcap << ",station=" << station << ",sector=" << sector << ",subsector=" << subSector
              << ",cscid=" << cscId << ")";
        }

      }  // MPC link loop

    //         std::vector<CSCSP_MBblock> mbStubs = sp->record(tbin).mbStubs();
    //         for(std::vector<CSCSP_MBblock>::const_iterator iter=mbStubs.begin(); iter!=mbStubs.end(); iter++){
    //           int endcap, sector;
    //           if( slot2sector[sp->header().slot()] ){
    //             endcap = slot2sector[sp->header().slot()]/7 + 1;
    //             sector = slot2sector[sp->header().slot()];
    //             if( sector>6 ) sector -= 6;
    //           } else {
    //             endcap = (sp->header().endcap()?1:2);
    //             sector =  sp->header().sector();
    //           }
    //           const unsigned int csc2dt[6][2] = {{2,3},{4,5},{6,7},{8,9},{10,11},{12,1}};
    //           DTChamberId id((endcap==1?2:-2),1, csc2dt[sector-1][iter->id()-1]);
    //           CSCCorrelatedLCTDigi base(0,iter->vq(),iter->quality(),iter->cal(),iter->flag(),iter->bc0(),iter->phi_bend(),tbin+(central_lct_bx-central_sp_bx),iter->id(),iter->bxn(),iter->timingError(),iter->BXN());
    //           csctf::TrackStub dtStub(base,id,iter->phi(),0);
    //           dtProduct->push_back(dtStub);
    //         }

  }  // tbin loop

  strip++;

  //e.put(std::move(dtProduct),"DT");
  e.put(std::move(LCTProduct));  // put processed lcts into the event.
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CSCTFSingleGen);
