#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include <string>

class L1TdeStage2CPPF : public DQMEDAnalyzer {
public:
  L1TdeStage2CPPF(const edm::ParameterSet& ps);
  ~L1TdeStage2CPPF() override;

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  int occupancy_value(int region_, int station_, int ring_);
  int bx_value(int region_, int emtfsector_);
  int GetSubsector(int emtfsector_, int lsubsector_);

  edm::EDGetTokenT<l1t::CPPFDigiCollection> dataToken;
  edm::EDGetTokenT<l1t::CPPFDigiCollection> emulToken;
  std::string monitorDir;
  bool verbose;

  MonitorElement* h2_Matching_SameKey_OnPhi_phi_Ce_phi_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OnPhi_theta_Ce_theta_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OnPhi_zone_Ce_zone_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OnPhi_ID_Ce_ID_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OnPhi_ID_Ce_roll_Ce_bx;

  MonitorElement* h2_Matching_SameKey_OffPhi_phi_Ce_phi_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OffPhi_theta_Ce_theta_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OffPhi_zone_Ce_zone_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OffPhi_ID_Ce_ID_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OffPhi_ID_Ce_roll_Ce_bx;

  MonitorElement* h2_Matching_SameKey_OnTheta_phi_Ce_phi_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OnTheta_theta_Ce_theta_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OnTheta_zone_Ce_zone_Cu_bx;

  MonitorElement* h2_Matching_SameKey_OffTheta_phi_Ce_phi_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OffTheta_theta_Ce_theta_Cu_bx;
  MonitorElement* h2_Matching_SameKey_OffTheta_zone_Ce_zone_Cu_bx;

  MonitorElement* h1_Matching_SameKey_bx_Summary;
};

L1TdeStage2CPPF::L1TdeStage2CPPF(const edm::ParameterSet& ps)
    : dataToken(consumes<l1t::CPPFDigiCollection>(ps.getParameter<edm::InputTag>("dataSource"))),
      emulToken(consumes<l1t::CPPFDigiCollection>(ps.getParameter<edm::InputTag>("emulSource"))),
      monitorDir(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose(ps.getUntrackedParameter<bool>("verbose", false)) {}

L1TdeStage2CPPF::~L1TdeStage2CPPF() {}

void L1TdeStage2CPPF::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  ibooker.setCurrentFolder(monitorDir);

  h2_Matching_SameKey_OnPhi_phi_Ce_phi_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OnPhi_phi_Ce_phi_Cu_bx",
                     "Matching && Same SubSector && OnPhi ; Emulator #phi ; Unpacker #phi",
                     62,
                     0.0,
                     1240.,
                     62,
                     0.0,
                     1240.);
  h2_Matching_SameKey_OnPhi_theta_Ce_theta_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OnPhi_theta_Ce_theta_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OnPhi ; Emulator #theta ; Unpacker #theta ",
                     32,
                     0,
                     32.,
                     32,
                     0,
                     32.);
  h2_Matching_SameKey_OnPhi_zone_Ce_zone_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OnPhi_zone_Ce_zone_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OnPhi ;Emulator Zone ;Unpacker Zone ",
                     15,
                     0,
                     15,
                     15,
                     0,
                     15);
  h2_Matching_SameKey_OnPhi_ID_Ce_ID_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OnPhi_ID_Ce_ID_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OnPhi ; Emulator Chamber ID;Unpacker Chamber ID ",
                     38,
                     0,
                     38,
                     38,
                     0,
                     38);
  h2_Matching_SameKey_OnPhi_ID_Ce_roll_Ce_bx =
      ibooker.book2D("h2_Matching_SameKey_OnPhi_ID_Ce_roll_Ce_bx",
                     "Matching && Same SubSector && bx==0 && OnPhi ;Emulator Chamber ID ;Emulator Roll ",
                     38,
                     0,
                     38,
                     4,
                     0,
                     4);

  h2_Matching_SameKey_OffPhi_phi_Ce_phi_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OffPhi_phi_Ce_phi_Cu_bx",
                     "Matching && Same SubSector && OffPhi ; Emulator #phi ; Unpacker #phi",
                     62,
                     0.0,
                     1240.,
                     62,
                     0.0,
                     1240.);
  h2_Matching_SameKey_OffPhi_theta_Ce_theta_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OffPhi_theta_Ce_theta_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OffPhi ; Emulator #theta ; Unpacker #theta ",
                     32,
                     0,
                     32.,
                     32,
                     0,
                     32.);
  h2_Matching_SameKey_OffPhi_zone_Ce_zone_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OffPhi_zone_Ce_zone_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OffPhi ;Emulator Zone ;Unpacker Zone ",
                     15,
                     0,
                     15,
                     15,
                     0,
                     15);
  h2_Matching_SameKey_OffPhi_ID_Ce_ID_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OffPhi_ID_Ce_ID_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OffPhi ; Emulator Chamber ID;Unpacker Chamber ID ",
                     38,
                     0,
                     38,
                     38,
                     0,
                     38);
  h2_Matching_SameKey_OffPhi_ID_Ce_roll_Ce_bx =
      ibooker.book2D("h2_Matching_SameKey_OffPhi_ID_Ce_roll_Ce_bx",
                     "Matching && Same SubSector && bx==0 && OffPhi ;Emulator Chamber ID ;Emulator Roll ",
                     38,
                     0,
                     38,
                     4,
                     0,
                     4);

  h2_Matching_SameKey_OnTheta_phi_Ce_phi_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OnTheta_phi_Ce_phi_Cu_bx",
                     "Matching && Same SubSector && OnTheta ; Emulator #phi ; Unpacker #phi",
                     62,
                     0.0,
                     1240.,
                     62,
                     0.0,
                     1240.);
  h2_Matching_SameKey_OnTheta_theta_Ce_theta_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OnTheta_theta_Ce_theta_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OnTheta ; Emulator #theta ; Unpacker #theta ",
                     32,
                     0,
                     32.,
                     32,
                     0,
                     32.);
  h2_Matching_SameKey_OnTheta_zone_Ce_zone_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OnTheta_zone_Ce_zone_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OnTheta ;Emulator Zone ;Unpacker Zone ",
                     15,
                     0,
                     15,
                     15,
                     0,
                     15);

  h2_Matching_SameKey_OffTheta_phi_Ce_phi_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OffTheta_phi_Ce_phi_Cu_bx",
                     "Matching && Same SubSector && OffTheta ; Emulator #phi ; Unpacker #phi",
                     62,
                     0.0,
                     1240.,
                     62,
                     0.0,
                     1240.);
  h2_Matching_SameKey_OffTheta_theta_Ce_theta_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OffTheta_theta_Ce_theta_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OffTheta ; Emulator #theta ; Unpacker #theta ",
                     32,
                     0,
                     32.,
                     32,
                     0,
                     32.);
  h2_Matching_SameKey_OffTheta_zone_Ce_zone_Cu_bx =
      ibooker.book2D("h2_Matching_SameKey_OffTheta_zone_Ce_zone_Cu_bx",
                     "Matching && Same SubSector && bx==0 && OffTheta ;Emulator Zone ;Unpacker Zone ",
                     15,
                     0,
                     15,
                     15,
                     0,
                     15);

  h1_Matching_SameKey_bx_Summary =
      ibooker.book1D("h1_Matching_SameKey_bx_Summary",
                     "cppf data-emul mismatch fraction summary; ; Fraction events with mismatch",
                     2,
                     1,
                     3);
  h1_Matching_SameKey_bx_Summary->setBinLabel(1, "off/on-phi");
  h1_Matching_SameKey_bx_Summary->setBinLabel(2, "off/on-theta");
  h1_Matching_SameKey_bx_Summary->setAxisRange(0, 0.01, 2);
}

void L1TdeStage2CPPF::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose)
    edm::LogInfo("L1TdeStage2CPPF") << "L1TdeStage2CPPF: analyze...";

  edm::Handle<l1t::CPPFDigiCollection> dataCPPFs;
  e.getByToken(dataToken, dataCPPFs);

  edm::Handle<l1t::CPPFDigiCollection> emulCPPFs;
  e.getByToken(emulToken, emulCPPFs);

  std::unordered_map<int, int> _nHit_Ce;
  std::unordered_map<int, int> _nHit_Cu;
  std::unordered_map<int, std::vector<int>> _phi_Ce;
  std::unordered_map<int, std::vector<int>> _phi_Cu;
  std::unordered_map<int, std::vector<int>> _phi_glob_Ce;
  std::unordered_map<int, std::vector<int>> _phi_glob_Cu;
  std::unordered_map<int, std::vector<int>> _theta_Ce;
  std::unordered_map<int, std::vector<int>> _theta_Cu;
  std::unordered_map<int, std::vector<int>> _theta_glob_Ce;
  std::unordered_map<int, std::vector<int>> _theta_glob_Cu;
  std::unordered_map<int, std::vector<int>> _roll_Ce;
  std::unordered_map<int, std::vector<int>> _roll_Cu;
  std::unordered_map<int, std::vector<int>> _zone_Ce;
  std::unordered_map<int, std::vector<int>> _zone_Cu;
  std::unordered_map<int, std::vector<int>> _ID_Ce;
  std::unordered_map<int, std::vector<int>> _ID_Cu;
  std::unordered_map<int, std::vector<int>> _emtfSubsector_Ce;
  std::unordered_map<int, std::vector<int>> _emtfSubsector_Cu;
  std::unordered_map<int, std::vector<int>> _emtfSector_Ce;
  std::unordered_map<int, std::vector<int>> _emtfSector_Cu;
  std::unordered_map<int, std::vector<int>> _bx_Ce;
  std::unordered_map<int, std::vector<int>> _bx_Cu;
  std::unordered_map<int, std::vector<int>> _cluster_size_Ce;
  std::unordered_map<int, std::vector<int>> _cluster_size_Cu;

  for (auto& cppf_digis : *emulCPPFs) {
    RPCDetId rpcIdCe = (int)cppf_digis.rpcId();
    int regionCe = (int)rpcIdCe.region();
    int stationCe = (int)rpcIdCe.station();
    int sectorCe = (int)rpcIdCe.sector();
    int subsectorCe = (int)rpcIdCe.subsector();
    int ringCe = (int)rpcIdCe.ring();
    int rollCe = (int)(rpcIdCe.roll());
    int phiIntCe = (int)cppf_digis.phi_int();
    int thetaIntCe = (int)cppf_digis.theta_int();
    int phiGlobalCe = (int)cppf_digis.phi_glob();
    int thetaGlobalCe = (int)cppf_digis.theta_glob();
    int cluster_sizeCe = (int)cppf_digis.cluster_size();
    int bxCe = cppf_digis.bx();
    int emtfSectorCe = (int)cppf_digis.emtf_sector();
    int emtfSubsectorCe = GetSubsector(emtfSectorCe, subsectorCe);
    int fillOccupancyCe = occupancy_value(regionCe, stationCe, ringCe);

    int nsubCe = 6;
    (ringCe == 1 && stationCe > 1) ? nsubCe = 3 : nsubCe = 6;
    int chamberIDCe = subsectorCe + nsubCe * (sectorCe - 1);

    std::ostringstream oss;
    oss << regionCe << stationCe << ringCe << sectorCe << subsectorCe << emtfSectorCe << emtfSubsectorCe;
    std::istringstream iss(oss.str());
    int unique_id;
    iss >> unique_id;

    if (_nHit_Ce.find(unique_id) == _nHit_Ce.end()) {
      _nHit_Ce.insert({unique_id, 1});
      _phi_Ce[unique_id].push_back(phiIntCe);
      _phi_glob_Ce[unique_id].push_back(phiGlobalCe);
      _theta_Ce[unique_id].push_back(thetaIntCe);
      _theta_glob_Ce[unique_id].push_back(thetaGlobalCe);
      _roll_Ce[unique_id].push_back(rollCe);
      _ID_Ce[unique_id].push_back(chamberIDCe);
      _zone_Ce[unique_id].push_back(fillOccupancyCe);
      _emtfSubsector_Ce[unique_id].push_back(emtfSubsectorCe);
      _emtfSector_Ce[unique_id].push_back(emtfSectorCe);
      _bx_Ce[unique_id].push_back(bxCe);
      _cluster_size_Ce[unique_id].push_back(cluster_sizeCe);
    } else {
      _nHit_Ce.at(unique_id) += 1;
      _phi_Ce[unique_id].push_back(phiIntCe);
      _phi_glob_Ce[unique_id].push_back(phiGlobalCe);
      _theta_Ce[unique_id].push_back(thetaIntCe);
      _theta_glob_Ce[unique_id].push_back(thetaGlobalCe);
      _roll_Ce[unique_id].push_back(rollCe);
      _ID_Ce[unique_id].push_back(chamberIDCe);
      _zone_Ce[unique_id].push_back(fillOccupancyCe);
      _emtfSubsector_Ce[unique_id].push_back(emtfSubsectorCe);
      _emtfSector_Ce[unique_id].push_back(emtfSectorCe);
      _bx_Ce[unique_id].push_back(bxCe);
      _cluster_size_Ce[unique_id].push_back(cluster_sizeCe);
    }

  }  // END :: for(auto& cppf_digis : *CppfDigis1)

  for (auto& cppf_digis2 : *dataCPPFs) {
    RPCDetId rpcIdCu = cppf_digis2.rpcId();
    int regionCu = (int)rpcIdCu.region();
    int stationCu = (int)rpcIdCu.station();
    int sectorCu = (int)rpcIdCu.sector();
    int subsectorCu = (int)rpcIdCu.subsector();
    int ringCu = (int)rpcIdCu.ring();
    int rollCu = (int)(rpcIdCu.roll());
    int phiIntCu = (int)cppf_digis2.phi_int();
    int thetaIntCu = (int)cppf_digis2.theta_int();
    int phiGlobalCu = (int)cppf_digis2.phi_glob();
    int thetaGlobalCu = (int)cppf_digis2.theta_glob();
    int cluster_sizeCu = (int)cppf_digis2.cluster_size();
    int bxCu = (int)cppf_digis2.bx();
    int emtfSectorCu = (int)cppf_digis2.emtf_sector();
    int emtfSubsectorCu = GetSubsector(emtfSectorCu, subsectorCu);
    int fillOccupancyCu = occupancy_value(regionCu, stationCu, ringCu);

    int nsubCu = 6;
    (ringCu == 1 && stationCu > 1) ? nsubCu = 3 : nsubCu = 6;
    int chamberIDCu = subsectorCu + nsubCu * (sectorCu - 1);

    std::ostringstream oss2;
    oss2 << regionCu << stationCu << ringCu << sectorCu << subsectorCu << emtfSectorCu << emtfSubsectorCu;
    std::istringstream iss2(oss2.str());
    int unique_id;
    iss2 >> unique_id;

    if (_nHit_Cu.find(unique_id) == _nHit_Cu.end()) {  // chamber had no hit so far
      _nHit_Cu.insert({unique_id, 1});
      _phi_Cu[unique_id].push_back(phiIntCu);
      _theta_Cu[unique_id].push_back(thetaIntCu);
      _phi_glob_Cu[unique_id].push_back(phiGlobalCu);
      _theta_glob_Cu[unique_id].push_back(thetaGlobalCu);
      _ID_Cu[unique_id].push_back(chamberIDCu);
      _zone_Cu[unique_id].push_back(fillOccupancyCu);
      _roll_Cu[unique_id].push_back(rollCu);
      _emtfSubsector_Cu[unique_id].push_back(emtfSubsectorCu);
      _emtfSector_Cu[unique_id].push_back(emtfSectorCu);
      _bx_Cu[unique_id].push_back(bxCu);
      _cluster_size_Cu[unique_id].push_back(cluster_sizeCu);
    } else {
      _nHit_Cu.at(unique_id) += 1;
      _phi_Cu[unique_id].push_back(phiIntCu);
      _theta_Cu[unique_id].push_back(thetaIntCu);
      _phi_glob_Cu[unique_id].push_back(phiGlobalCu);
      _theta_glob_Cu[unique_id].push_back(thetaGlobalCu);
      _ID_Cu[unique_id].push_back(chamberIDCu);
      _zone_Cu[unique_id].push_back(fillOccupancyCu);
      _roll_Cu[unique_id].push_back(rollCu);
      _emtfSubsector_Cu[unique_id].push_back(emtfSubsectorCu);
      _emtfSector_Cu[unique_id].push_back(emtfSectorCu);
      _bx_Cu[unique_id].push_back(bxCu);
      _cluster_size_Cu[unique_id].push_back(cluster_sizeCu);
    }
  }  // END: : for(auto& cppf_digis2 : *CppfDigis2)

  for (auto const& Ce : _nHit_Ce) {
    int key_Ce = Ce.first;
    int nHit_Ce = Ce.second;

    for (auto const& Cu : _nHit_Cu) {
      int key_Cu = Cu.first;
      int nHit_Cu = Cu.second;

      if (key_Ce != key_Cu)
        continue;
      if (nHit_Ce != nHit_Cu)
        continue;

      for (int vecSize = 0; vecSize < nHit_Cu; ++vecSize) {
        if (_bx_Cu.at(key_Cu)[vecSize] != _bx_Ce.at(key_Ce)[vecSize])
          continue;

        bool OnPhi_Matching = false;
        int index_Ce = vecSize;
        int index_Cu = vecSize;
        for (int i = 0; i < nHit_Ce; ++i) {
          if (_phi_Ce.at(key_Ce)[i] == _phi_Cu.at(key_Cu)[vecSize]) {
            OnPhi_Matching = true;
            index_Cu = vecSize;
            index_Ce = i;
          }
        }
        if (OnPhi_Matching) {
          h2_Matching_SameKey_OnPhi_phi_Ce_phi_Cu_bx->Fill(_phi_Ce.at(key_Ce)[index_Ce], _phi_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OnPhi_theta_Ce_theta_Cu_bx->Fill(_theta_Ce.at(key_Ce)[index_Ce],
                                                               _theta_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OnPhi_zone_Ce_zone_Cu_bx->Fill(_zone_Ce.at(key_Ce)[index_Ce],
                                                             _zone_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OnPhi_ID_Ce_ID_Cu_bx->Fill(_ID_Ce.at(key_Ce)[index_Ce], _ID_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OnPhi_ID_Ce_roll_Ce_bx->Fill(_ID_Ce.at(key_Ce)[index_Ce], _roll_Ce.at(key_Ce)[index_Ce]);
        } else {
          h2_Matching_SameKey_OffPhi_phi_Ce_phi_Cu_bx->Fill(_phi_Ce.at(key_Ce)[index_Ce], _phi_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OffPhi_theta_Ce_theta_Cu_bx->Fill(_theta_Ce.at(key_Ce)[index_Ce],
                                                                _theta_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OffPhi_zone_Ce_zone_Cu_bx->Fill(_zone_Ce.at(key_Ce)[index_Ce],
                                                              _zone_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OffPhi_ID_Ce_ID_Cu_bx->Fill(_ID_Ce.at(key_Ce)[index_Ce], _ID_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OffPhi_ID_Ce_roll_Ce_bx->Fill(_ID_Ce.at(key_Ce)[index_Ce], _roll_Ce.at(key_Ce)[index_Ce]);
        }

        bool OnTheta_Matching = false;
        for (int i = 0; i < nHit_Ce; ++i) {
          if (_theta_Ce.at(key_Ce)[i] == _theta_Cu.at(key_Cu)[index_Cu]) {
            OnTheta_Matching = true;
            index_Cu = vecSize;
            index_Ce = i;
          }
        }
        if (OnTheta_Matching) {
          h2_Matching_SameKey_OnTheta_phi_Ce_phi_Cu_bx->Fill(_phi_Ce.at(key_Ce)[index_Ce],
                                                             _phi_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OnTheta_theta_Ce_theta_Cu_bx->Fill(_theta_Ce.at(key_Ce)[index_Ce],
                                                                 _theta_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OnTheta_zone_Ce_zone_Cu_bx->Fill(_zone_Ce.at(key_Ce)[index_Ce],
                                                               _zone_Cu.at(key_Cu)[index_Cu]);
        } else {
          h2_Matching_SameKey_OffTheta_phi_Ce_phi_Cu_bx->Fill(_phi_Ce.at(key_Ce)[index_Ce],
                                                              _phi_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OffTheta_theta_Ce_theta_Cu_bx->Fill(_theta_Ce.at(key_Ce)[index_Ce],
                                                                  _theta_Cu.at(key_Cu)[index_Cu]);
          h2_Matching_SameKey_OffTheta_zone_Ce_zone_Cu_bx->Fill(_zone_Ce.at(key_Ce)[index_Ce],
                                                                _zone_Cu.at(key_Cu)[index_Cu]);
        }
      }

      double off_phi, on_phi, off_theta, on_theta;
      on_phi = h2_Matching_SameKey_OnPhi_phi_Ce_phi_Cu_bx->getEntries();
      off_phi = h2_Matching_SameKey_OffPhi_phi_Ce_phi_Cu_bx->getEntries();
      on_theta = h2_Matching_SameKey_OnTheta_theta_Ce_theta_Cu_bx->getEntries();
      off_theta = h2_Matching_SameKey_OffTheta_theta_Ce_theta_Cu_bx->getEntries();
      if (on_phi == 0)
        on_phi = 0.0001;
      if (on_theta == 0)
        on_theta = 0.0001;
      h1_Matching_SameKey_bx_Summary->setBinContent(1, off_phi / on_phi);
      h1_Matching_SameKey_bx_Summary->setBinContent(2, off_theta / on_theta);
    }
  }
}

int L1TdeStage2CPPF::GetSubsector(int emtfsector_, int lsubsector_) {
  const int nsectors = 6;
  int gsubsector = 0;
  if ((emtfsector_ != -99) and (lsubsector_ != 0)) {
    gsubsector = (emtfsector_ - 1) * nsectors + lsubsector_;
  }
  return gsubsector;
}

int L1TdeStage2CPPF::occupancy_value(int region_, int station_, int ring_) {
  int fill_val = 0;
  if (region_ == -1) {
    if ((station_ == 4) && (ring_ == 3))
      fill_val = 1;
    else if ((station_ == 4) && (ring_ == 2))
      fill_val = 2;
    else if ((station_ == 3) && (ring_ == 3))
      fill_val = 3;
    else if ((station_ == 3) && (ring_ == 2))
      fill_val = 4;
    else if ((station_ == 2) && (ring_ == 2))
      fill_val = 5;
    else if ((station_ == 1) && (ring_ == 2))
      fill_val = 6;

  } else if (region_ == +1) {
    if ((station_ == 1) && (ring_ == 2))
      fill_val = 7;
    else if ((station_ == 2) && (ring_ == 2))
      fill_val = 8;
    else if ((station_ == 3) && (ring_ == 2))
      fill_val = 9;
    else if ((station_ == 3) && (ring_ == 3))
      fill_val = 10;
    else if ((station_ == 4) && (ring_ == 2))
      fill_val = 11;
    else if ((station_ == 4) && (ring_ == 3))
      fill_val = 12;
  }
  return fill_val;
}

int L1TdeStage2CPPF::bx_value(int region_, int emtfsector_) {
  int fill_val = 0;

  if (region_ == -1) {
    fill_val = 7 - emtfsector_;
  } else if (region_ == +1) {
    fill_val = 6 + emtfsector_;
  }
  return fill_val;
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TdeStage2CPPF);
