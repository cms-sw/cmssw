#include <vector>
#include <string>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"

// class decleration

class L1TStage2CPPF : public DQMEDAnalyzer {
public:
  // class constructor
  L1TStage2CPPF(const edm::ParameterSet& ps);
  // class destructor
  ~L1TStage2CPPF() override;

  // member functions
  edm::ESHandle<RPCGeometry> rpcGeom;

protected:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;

  // data members
private:
  std::string monitorDir;
  bool verbose;
  float global_phi;
  const edm::EDGetTokenT<l1t::CPPFDigiCollection> cppfDigiToken_;
  int EMTF_sector;
  int EMTF_subsector;
  int EMTF_bx;

  std::vector<int> EMTFsector1bins;
  std::vector<int> EMTFsector2bins;
  std::vector<int> EMTFsector3bins;
  std::vector<int> EMTFsector4bins;
  std::vector<int> EMTFsector5bins;
  std::vector<int> EMTFsector6bins;

  std::map<int, std::vector<int>> fill_info;

  MonitorElement* Occupancy_EMTFSector;
  MonitorElement* Track_Bx;
};

L1TStage2CPPF::L1TStage2CPPF(const edm::ParameterSet& ps)
    : monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)),
      global_phi(-1000),
      cppfDigiToken_(consumes<l1t::CPPFDigiCollection>(ps.getParameter<edm::InputTag>("cppfSource"))) {}

L1TStage2CPPF::~L1TStage2CPPF() {}

void L1TStage2CPPF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run& iRun, const edm::EventSetup& eveSetup) {
  ibooker.setCurrentFolder(monitorDir);

  Occupancy_EMTFSector = ibooker.book2D("Occupancy_EMTFSector", "Occupancy_EMTFSector", 36, 1., 37., 12, 1., 13.);
  Track_Bx = ibooker.book2D("Track_Bx", "Track_Bx", 12, 1., 13., 7, -3., 4.);
}

void L1TStage2CPPF::analyze(const edm::Event& eve, const edm::EventSetup& eveSetup) {
  if (verbose) {
    edm::LogInfo("L1TStage2CPPF") << "L1TStage2CPPF: analyze....";
  }

  edm::Handle<l1t::CPPFDigiCollection> CppfDigis;
  eve.getByToken(cppfDigiToken_, CppfDigis);

  //Fill the specific bin for each EMTF sector
  EMTFsector1bins.clear();
  EMTFsector2bins.clear();
  EMTFsector3bins.clear();
  EMTFsector4bins.clear();
  EMTFsector5bins.clear();
  EMTFsector6bins.clear();
  for (int i = 1; i < 7; i++) {
    EMTFsector1bins.push_back(i);
    EMTFsector2bins.push_back(i + 6);
    EMTFsector3bins.push_back(i + 12);
    EMTFsector4bins.push_back(i + 18);
    EMTFsector5bins.push_back(i + 24);
    EMTFsector6bins.push_back(i + 30);
  }
  //FIll the map for each EMTF sector
  fill_info[1] = EMTFsector1bins;
  fill_info[2] = EMTFsector2bins;
  fill_info[3] = EMTFsector3bins;
  fill_info[4] = EMTFsector4bins;
  fill_info[5] = EMTFsector5bins;
  fill_info[6] = EMTFsector6bins;

  for (auto& cppf_digis : *CppfDigis) {
    RPCDetId rpcId = cppf_digis.rpcId();
    int ring = rpcId.ring();
    int station = rpcId.station();
    int region = rpcId.region();
    int subsector = rpcId.subsector();

    //Region -
    if (region == -1) {
      //for Occupancy
      EMTF_sector = rpcId.sector();
      EMTF_subsector = fill_info[EMTF_sector][subsector - 1];

      if ((station == 4) && (ring == 3)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 1);
      } else if ((station == 4) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 2);
      } else if ((station == 3) && (ring == 3)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 3);
      } else if ((station == 3) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 4);
      } else if ((station == 2) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 5);
      } else if ((station == 1) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 6);
      }

      //for Track_Bx
      if (EMTF_sector >= 1 && EMTF_sector <= 6) {
        EMTF_bx = cppf_digis.bx();
        Track_Bx->Fill(7 - EMTF_sector, EMTF_bx);
      }
    }
    //Region +
    if (region == 1) {
      //for Occupancy
      EMTF_sector = rpcId.sector();
      EMTF_subsector = fill_info[EMTF_sector][subsector - 1];

      if ((station == 1) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 7);
      } else if ((station == 2) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 8);
      } else if ((station == 3) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 9);
      } else if ((station == 3) && (ring == 3)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 10);
      } else if ((station == 4) && (ring == 2)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 11);
      } else if ((station == 4) && (ring == 3)) {
        Occupancy_EMTFSector->Fill(EMTF_subsector, 12);
      }

      //for Track_Bx
      if (EMTF_sector >= 1 && EMTF_sector <= 6) {
        EMTF_bx = cppf_digis.bx();
        Track_Bx->Fill(6 + EMTF_sector, EMTF_bx);
      }
    }
  }  //loop over CPPFDigis
}
DEFINE_FWK_MODULE(L1TStage2CPPF);
