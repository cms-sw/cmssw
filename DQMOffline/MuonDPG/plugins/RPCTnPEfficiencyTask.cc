/*
 * \file RPCTnPEfficiencyTask.cc
 *
 * \author L. Lunerti - INFN Bologna
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/MuonSegmentMatch.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "DQMOffline/MuonDPG/interface/BaseTnPEfficiencyTask.h"

class RPCTnPEfficiencyTask : public BaseTnPEfficiencyTask {
public:
  /// Constructor
  RPCTnPEfficiencyTask(const edm::ParameterSet& config);

  /// Destructor
  ~RPCTnPEfficiencyTask() override;

protected:
  std::string topFolder() const override;

  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& context) override;

  /// Book wheel granularity histograms
  void bookWheelHistos(DQMStore::IBooker& iBooker, int wheel, std::string folder = "");

  /// Book endcap histograms
  void bookEndcapHistos(DQMStore::IBooker& iBooker, int stations, std::string folder = "");

  int get_barrel_histo_ycoord(int ring, int station, int sector, int layer, int subsector, int roll);

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context) override;
};

RPCTnPEfficiencyTask::RPCTnPEfficiencyTask(const edm::ParameterSet& config) : BaseTnPEfficiencyTask(config) {
  LogTrace("DQMOffline|MuonDPG|RPCTnPEfficiencyTask") << "[RPCTnPEfficiencyTask]: Constructor" << std::endl;
}

RPCTnPEfficiencyTask::~RPCTnPEfficiencyTask() {
  LogTrace("DQMOffline|MuonDPG|RPCTnPEfficiencyTask")
      << "[RPCTnPEfficiencyTask]: analyzed " << m_nEvents << " events" << std::endl;
}

void RPCTnPEfficiencyTask::bookHistograms(DQMStore::IBooker& iBooker,
                                          edm::Run const& run,
                                          edm::EventSetup const& context) {
  BaseTnPEfficiencyTask::bookHistograms(iBooker, run, context);

  LogTrace("DQMOffline|MuonDPG|RPCTnPEfficiencyTask") << "[RPCTnPEfficiencyTask]: bookHistograms" << std::endl;

  for (int wheel = -2; wheel <= 2; ++wheel) {
    bookWheelHistos(iBooker, wheel, "Task");
  }

  for (int station = -4; station <= 4; ++station) {
    if (station == 0)
      continue;
    bookEndcapHistos(iBooker, station, "Task");
  }

  auto baseDir = topFolder() + "Task/";
  iBooker.setCurrentFolder(baseDir);

  MonitorElement* me_RPC_barrel_pass_allCh_1D =
      iBooker.book1D("RPC_nPassingProbe_Barrel_allCh_1D", "RPC_nPassingProbe_Barrel_allCh_1D", 20, 0.5, 20.5);
  MonitorElement* me_RPC_barrel_fail_allCh_1D =
      iBooker.book1D("RPC_nFailingProbe_Barrel_allCh_1D", "RPC_nFailingProbe_Barrel_allCh_1D", 20, 0.5, 20.5);

  MonitorElement* me_RPC_endcap_pass_allCh_1D =
      iBooker.book1D("RPC_nPassingProbe_Endcap_allCh_1D", "RPC_nPassingProbe_Endcap_allCh_1D", 9, -4., 5.);
  MonitorElement* me_RPC_endcap_fail_allCh_1D =
      iBooker.book1D("RPC_nFailingProbe_Endcap_allCh_1D", "RPC_nFailingProbe_Endcap_allCh_1D", 9, -4., 5.);

  me_RPC_barrel_pass_allCh_1D->setBinLabel(1, "RB1/YB-2", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(2, "RB2/YB-2", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(3, "RB3/YB-2", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(4, "RB4/YB-2", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(5, "RB1/YB-1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(6, "RB2/YB-1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(7, "RB3/YB-1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(8, "RB4/YB-1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(9, "RB1/YB0", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(10, "RB2/YB0", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(11, "RB3/YB0", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(12, "RB4/YB0", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(13, "RB1/YB1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(14, "RB2/YB1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(15, "RB3/YB1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(16, "RB4/YB1", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(17, "RB1/YB2", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(18, "RB2/YB2", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(19, "RB3/YB2", 1);
  me_RPC_barrel_pass_allCh_1D->setBinLabel(20, "RB4/YB2", 1);
  me_RPC_barrel_pass_allCh_1D->setAxisTitle("Number of passing probes", 2);

  me_RPC_barrel_fail_allCh_1D->setBinLabel(1, "RB1/YB-2", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(2, "RB2/YB-2", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(3, "RB3/YB-2", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(4, "RB4/YB-2", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(5, "RB1/YB-1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(6, "RB2/YB-1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(7, "RB3/YB-1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(8, "RB4/YB-1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(9, "RB1/YB0", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(10, "RB2/YB0", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(11, "RB3/YB0", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(12, "RB4/YB0", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(13, "RB1/YB1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(14, "RB2/YB1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(15, "RB3/YB1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(16, "RB4/YB1", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(17, "RB1/YB2", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(18, "RB2/YB2", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(19, "RB3/YB2", 1);
  me_RPC_barrel_fail_allCh_1D->setBinLabel(20, "RB4/YB2", 1);
  me_RPC_barrel_fail_allCh_1D->setAxisTitle("Number of failing probes", 2);

  me_RPC_endcap_pass_allCh_1D->setBinLabel(1, "RE-4", 1);
  me_RPC_endcap_pass_allCh_1D->setBinLabel(2, "RE-3", 1);
  me_RPC_endcap_pass_allCh_1D->setBinLabel(3, "RE-2", 1);
  me_RPC_endcap_pass_allCh_1D->setBinLabel(4, "RE-1", 1);
  me_RPC_endcap_pass_allCh_1D->setBinLabel(6, "RE1", 1);
  me_RPC_endcap_pass_allCh_1D->setBinLabel(7, "RE2", 1);
  me_RPC_endcap_pass_allCh_1D->setBinLabel(8, "RE3", 1);
  me_RPC_endcap_pass_allCh_1D->setBinLabel(9, "RE4", 1);
  me_RPC_endcap_pass_allCh_1D->setAxisTitle("Number of passing probes", 2);

  me_RPC_endcap_fail_allCh_1D->setBinLabel(1, "RE-4", 1);
  me_RPC_endcap_fail_allCh_1D->setBinLabel(2, "RE-3", 1);
  me_RPC_endcap_fail_allCh_1D->setBinLabel(3, "RE-2", 1);
  me_RPC_endcap_fail_allCh_1D->setBinLabel(4, "RE-1", 1);
  me_RPC_endcap_fail_allCh_1D->setBinLabel(6, "RE1", 1);
  me_RPC_endcap_fail_allCh_1D->setBinLabel(7, "RE2", 1);
  me_RPC_endcap_fail_allCh_1D->setBinLabel(8, "RE3", 1);
  me_RPC_endcap_fail_allCh_1D->setBinLabel(9, "RE4", 1);
  me_RPC_endcap_fail_allCh_1D->setAxisTitle("Number of failing probes", 2);

  m_histos["RPC_nPassingProbe_Barrel_allCh_1D"] = me_RPC_barrel_pass_allCh_1D;
  m_histos["RPC_nFailingProbe_Barrel_allCh_1D"] = me_RPC_barrel_fail_allCh_1D;

  m_histos["RPC_nPassingProbe_Endcap_allCh_1D"] = me_RPC_endcap_pass_allCh_1D;
  m_histos["RPC_nFailingProbe_Endcap_allCh_1D"] = me_RPC_endcap_fail_allCh_1D;
}

void RPCTnPEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& context) {
  BaseTnPEfficiencyTask::analyze(event, context);

  edm::Handle<reco::MuonCollection> muons;
  event.getByToken(m_muToken, muons);

  //RPC variables
  std::vector<std::vector<int>> probe_coll_RPC_region;
  std::vector<std::vector<int>> probe_coll_RPC_ring;
  std::vector<std::vector<int>> probe_coll_RPC_sta;
  std::vector<std::vector<int>> probe_coll_RPC_sec;
  std::vector<std::vector<int>> probe_coll_RPC_lay;
  std::vector<std::vector<int>> probe_coll_RPC_sub;
  std::vector<std::vector<int>> probe_coll_RPC_roll;
  std::vector<std::vector<float>> probe_coll_RPC_dx;
  std::vector<uint8_t> probe_coll_RPC_staMatch;

  std::vector<unsigned> probe_indices;
  if (!m_probeIndices.empty())
    probe_indices = m_probeIndices.back();

  //Fill probe dx + subdetector coordinates
  for (const auto i : probe_indices) {
    //RPC variables
    std::vector<int> probe_RPC_region;
    std::vector<int> probe_RPC_ring;
    std::vector<int> probe_RPC_sta;
    std::vector<int> probe_RPC_sec;
    std::vector<int> probe_RPC_lay;
    std::vector<int> probe_RPC_sub;
    std::vector<int> probe_RPC_roll;
    std::vector<float> probe_RPC_dx;
    uint8_t RPC_stationMatching = 0;

    float rpc_matched = false;  // fill detailed plots only for probes matching RPC

    for (const auto& chambMatch : (*muons).at(i).matches()) {
      // look in RPCs
      if (chambMatch.detector() == MuonSubdetId::RPC) {
        if (chambMatch.edgeX < m_borderCut && chambMatch.edgeY < m_borderCut) {
          rpc_matched = true;  //fill detailed plots if at least one RPC match

          RPCDetId chId(chambMatch.id.rawId());

          int region = chId.region();  // barrel if 0, endcap if -/+ 1
          int ring = chId.ring();      // means wheel in the barrel and ring in the endcap
          int station = chId.station();
          int sector = chId.sector();
          int subsector = chId.subsector();
          int layer = chId.layer();
          int roll = chId.roll();

          reco::MuonRPCHitMatch closest_matchedRPCHit;
          double smallestDx = 999.;
          for (auto& seg : chambMatch.rpcMatches) {
            float dx = std::abs(chambMatch.x - seg.x);

            if (dx < smallestDx) {
              smallestDx = dx;
              closest_matchedRPCHit = seg;
            }
          }

          RPC_stationMatching = RPC_stationMatching | (1 << (station - 1));

          probe_RPC_region.push_back(region);
          probe_RPC_ring.push_back(ring);
          probe_RPC_sta.push_back(station);
          probe_RPC_sec.push_back(sector);
          probe_RPC_lay.push_back(layer);
          probe_RPC_sub.push_back(subsector);
          probe_RPC_roll.push_back(roll);
          probe_RPC_dx.push_back(smallestDx);
        }
      } else
        continue;
    }  //loop over chamber matches

    //Fill detailed plots
    if (m_detailedAnalysis && rpc_matched) {
      m_histos.find("probeEta")->second->Fill((*muons).at(i).eta());
      m_histos.find("probePhi")->second->Fill((*muons).at(i).phi());
      m_histos.find("probeNumberOfMatchedStations")->second->Fill((*muons).at(i).numberOfMatchedStations());
      m_histos.find("probePt")->second->Fill((*muons).at(i).pt());
    }

    //Fill RPC variables
    probe_coll_RPC_region.push_back(probe_RPC_region);
    probe_coll_RPC_ring.push_back(probe_RPC_ring);
    probe_coll_RPC_sta.push_back(probe_RPC_sta);
    probe_coll_RPC_sec.push_back(probe_RPC_sec);
    probe_coll_RPC_lay.push_back(probe_RPC_lay);
    probe_coll_RPC_sub.push_back(probe_RPC_sub);
    probe_coll_RPC_roll.push_back(probe_RPC_roll);
    probe_coll_RPC_dx.push_back(probe_RPC_dx);
    probe_coll_RPC_staMatch.push_back(RPC_stationMatching);
  }  //loop over probe collection

  //Loop over probes
  for (unsigned i = 0; i < probe_indices.size(); ++i) {
    uint8_t RPC_matchPatt = probe_coll_RPC_staMatch.at(i);

    //Loop over RPC matches
    unsigned nRPC_matches = probe_coll_RPC_region.at(i).size();
    for (unsigned j = 0; j < nRPC_matches; ++j) {
      //RPC variables
      int RPC_region = probe_coll_RPC_region.at(i).at(j);
      int RPC_ring = probe_coll_RPC_ring.at(i).at(j);
      int RPC_sta = probe_coll_RPC_sta.at(i).at(j);
      int RPC_sec = probe_coll_RPC_sec.at(i).at(j);
      int RPC_lay = probe_coll_RPC_lay.at(i).at(j);
      int RPC_subsec = probe_coll_RPC_sub.at(i).at(j);
      int RPC_roll = probe_coll_RPC_roll.at(i).at(j);
      float RPC_dx = probe_coll_RPC_dx.at(i).at(j);

      //Fill RPC plots
      if ((RPC_matchPatt & (1 << (RPC_sta - 1))) != 0)  //avoids 0 station matching
      {
        if (RPC_dx < m_dxCut) {
          //Barrel region
          if (RPC_region == 0) {
            int barrel_histo_xcoord = RPC_sec;
            int barrel_histo_ycoord =
                get_barrel_histo_ycoord(RPC_ring, RPC_sta, RPC_sec, RPC_lay, RPC_subsec, RPC_roll);

            std::string hName = std::string("RPC_nPassingProbePerRoll_Barrel_W") + std::to_string(RPC_ring);
            m_histos.find(hName)->second->Fill(barrel_histo_xcoord, barrel_histo_ycoord);

            std::string hName_1D = std::string("RPC_nPassingProbePerRoll_Barrel_1D_W") + std::to_string(RPC_ring);
            m_histos.find(hName_1D)->second->Fill(barrel_histo_ycoord);

            m_histos.find("RPC_nPassingProbe_Barrel_allCh_1D")->second->Fill((RPC_sta) + 4 * (RPC_ring + 2));
          }
          //Endcap region
          else {
            int endcap_histo_xcoord = (6 * (RPC_sec - 1)) + RPC_subsec;
            int endcap_histo_ycoord = (3 * (RPC_ring - 2)) + RPC_roll;

            std::string hName =
                std::string("RPC_nPassingProbePerRoll_Endcap_Sta") + std::to_string(RPC_sta * RPC_region);
            m_histos.find(hName)->second->Fill(endcap_histo_xcoord, endcap_histo_ycoord);

            std::string hName_1D =
                std::string("RPC_nPassingProbePerRoll_Endcap_1D_Sta") + std::to_string(RPC_sta * RPC_region);
            m_histos.find(hName_1D)->second->Fill(endcap_histo_ycoord);

            m_histos.find("RPC_nPassingProbe_Endcap_allCh_1D")->second->Fill(RPC_region * RPC_sta);
          }
        } else {
          //Barrel region
          if (RPC_region == 0) {
            int barrel_histo_xcoord = RPC_sec;
            int barrel_histo_ycoord =
                get_barrel_histo_ycoord(RPC_ring, RPC_sta, RPC_sec, RPC_lay, RPC_subsec, RPC_roll);

            std::string hName = std::string("RPC_nFailingProbePerRoll_Barrel_W") + std::to_string(RPC_ring);
            m_histos.find(hName)->second->Fill(barrel_histo_xcoord, barrel_histo_ycoord);

            std::string hName_1D = std::string("RPC_nFailingProbePerRoll_Barrel_1D_W") + std::to_string(RPC_ring);
            m_histos.find(hName_1D)->second->Fill(barrel_histo_ycoord);

            m_histos.find("RPC_nFailingProbe_Barrel_allCh_1D")->second->Fill((RPC_sta) + 4 * (RPC_ring + 2));
          }
          //Endcap region
          else {
            int endcap_histo_xcoord = (6 * (RPC_sec - 1)) + RPC_subsec;
            int endcap_histo_ycoord = (3 * (RPC_ring - 2)) + RPC_roll;

            std::string hName =
                std::string("RPC_nFailingProbePerRoll_Endcap_Sta") + std::to_string(RPC_sta * RPC_region);
            m_histos.find(hName)->second->Fill(endcap_histo_xcoord, endcap_histo_ycoord);

            std::string hName_1D =
                std::string("RPC_nFailingProbePerRoll_Endcap_1D_Sta") + std::to_string(RPC_sta * RPC_region);
            m_histos.find(hName_1D)->second->Fill(endcap_histo_ycoord);

            m_histos.find("RPC_nFailingProbe_Endcap_allCh_1D")->second->Fill(RPC_region * RPC_sta);
          }
        }
      }
    }
  }
}

void RPCTnPEfficiencyTask::bookWheelHistos(DQMStore::IBooker& iBooker, int wheel, std::string folder) {
  auto baseDir = topFolder() + folder + "/";
  iBooker.setCurrentFolder(baseDir);

  LogTrace("DQMOffline|MuonDPG|RPCTnPEfficiencyTask")
      << "[RPCTnPEfficiencyTask]: booking histos in " << baseDir << std::endl;

  auto hName_RPC_pass = std::string("RPC_nPassingProbePerRoll_Barrel_W") + std::to_string(wheel);
  auto hName_RPC_fail = std::string("RPC_nFailingProbePerRoll_Barrel_W") + std::to_string(wheel);

  auto hName_RPC_pass_1D = std::string("RPC_nPassingProbePerRoll_Barrel_1D_W") + std::to_string(wheel);
  auto hName_RPC_fail_1D = std::string("RPC_nFailingProbePerRoll_Barrel_1D_W") + std::to_string(wheel);

  MonitorElement* me_RPC_pass =
      iBooker.book2D(hName_RPC_pass.c_str(), hName_RPC_pass.c_str(), 12, 0.5, 12.5, 21, 0., 21.5);
  MonitorElement* me_RPC_fail =
      iBooker.book2D(hName_RPC_fail.c_str(), hName_RPC_fail.c_str(), 12, 0.5, 12.5, 21, 0., 21.5);

  MonitorElement* me_RPC_pass_1D = iBooker.book1D(hName_RPC_pass_1D.c_str(), hName_RPC_pass_1D.c_str(), 21, 0., 21.5);
  MonitorElement* me_RPC_fail_1D = iBooker.book1D(hName_RPC_fail_1D.c_str(), hName_RPC_fail_1D.c_str(), 21, 0., 21.5);

  me_RPC_pass->setBinLabel(1, "RB1in B", 2);
  me_RPC_pass->setBinLabel(2, "RB1in F", 2);
  me_RPC_pass->setBinLabel(3, "RB1out B", 2);
  me_RPC_pass->setBinLabel(4, "RB1out F", 2);
  if (std::abs(wheel) < 2) {
    me_RPC_pass->setBinLabel(5, "RB2in B", 2);
    me_RPC_pass->setBinLabel(6, "RB2in M", 2);
    me_RPC_pass->setBinLabel(7, "RB2in F", 2);
    me_RPC_pass->setBinLabel(8, "RB2out B", 2);
    me_RPC_pass->setBinLabel(9, "RB2out F", 2);
  } else {
    me_RPC_pass->setBinLabel(5, "RB2in B", 2);
    me_RPC_pass->setBinLabel(6, "RB2in F", 2);
    me_RPC_pass->setBinLabel(7, "RB2out B", 2);
    me_RPC_pass->setBinLabel(8, "RB2out M", 2);
    me_RPC_pass->setBinLabel(9, "RB2out F", 2);
  }
  me_RPC_pass->setBinLabel(10, "RB3- B", 2);
  me_RPC_pass->setBinLabel(11, "RB3- F", 2);
  me_RPC_pass->setBinLabel(12, "RB3+ B", 2);
  me_RPC_pass->setBinLabel(13, "RB3+ F", 2);
  me_RPC_pass->setBinLabel(14, "RB4- B", 2);
  me_RPC_pass->setBinLabel(15, "RB4- F", 2);
  me_RPC_pass->setBinLabel(16, "RB4+ B", 2);
  me_RPC_pass->setBinLabel(17, "RB4+ F", 2);
  me_RPC_pass->setBinLabel(18, "RB4-- B", 2);
  me_RPC_pass->setBinLabel(19, "RB4-- F", 2);
  me_RPC_pass->setBinLabel(20, "RB4++ B", 2);
  me_RPC_pass->setBinLabel(21, "RB4++ F", 2);
  for (int i = 1; i < 13; ++i) {
    me_RPC_pass->setBinLabel(i, std::to_string(i), 1);
  }
  me_RPC_pass->setAxisTitle("Sector", 1);
  me_RPC_pass->setAxisTitle("Number of passing probes", 3);

  me_RPC_fail->setBinLabel(1, "RB1in B", 2);
  me_RPC_fail->setBinLabel(2, "RB1in F", 2);
  me_RPC_fail->setBinLabel(3, "RB1out B", 2);
  me_RPC_fail->setBinLabel(4, "RB1out F", 2);
  if (std::abs(wheel) < 2) {
    me_RPC_fail->setBinLabel(5, "RB2in B", 2);
    me_RPC_fail->setBinLabel(6, "RB2in M", 2);
    me_RPC_fail->setBinLabel(7, "RB2in F", 2);
    me_RPC_fail->setBinLabel(8, "RB2out B", 2);
    me_RPC_fail->setBinLabel(9, "RB2out F", 2);
  } else {
    me_RPC_fail->setBinLabel(5, "RB2in B", 2);
    me_RPC_fail->setBinLabel(6, "RB2in F", 2);
    me_RPC_fail->setBinLabel(7, "RB2out B", 2);
    me_RPC_fail->setBinLabel(8, "RB2out M", 2);
    me_RPC_fail->setBinLabel(9, "RB2out F", 2);
  }
  me_RPC_fail->setBinLabel(10, "RB3- B", 2);
  me_RPC_fail->setBinLabel(11, "RB3- F", 2);
  me_RPC_fail->setBinLabel(12, "RB3+ B", 2);
  me_RPC_fail->setBinLabel(13, "RB3+ F", 2);
  me_RPC_fail->setBinLabel(14, "RB4- B", 2);
  me_RPC_fail->setBinLabel(15, "RB4- F", 2);
  me_RPC_fail->setBinLabel(16, "RB4+ B", 2);
  me_RPC_fail->setBinLabel(17, "RB4+ F", 2);
  me_RPC_fail->setBinLabel(18, "RB4-- B", 2);
  me_RPC_fail->setBinLabel(19, "RB4-- F", 2);
  me_RPC_fail->setBinLabel(20, "RB4++ B", 2);
  me_RPC_fail->setBinLabel(21, "RB4++ F", 2);
  for (int i = 1; i < 13; ++i) {
    me_RPC_fail->setBinLabel(i, std::to_string(i), 1);
  }
  me_RPC_fail->setAxisTitle("Sector", 1);
  me_RPC_fail->setAxisTitle("Number of failing probes", 3);

  me_RPC_pass_1D->setBinLabel(1, "RB1in B", 1);
  me_RPC_pass_1D->setBinLabel(2, "RB1in F", 1);
  me_RPC_pass_1D->setBinLabel(3, "RB1out B", 1);
  me_RPC_pass_1D->setBinLabel(4, "RB1out F", 1);
  if (std::abs(wheel) < 2) {
    me_RPC_pass_1D->setBinLabel(5, "RB2in B", 1);
    me_RPC_pass_1D->setBinLabel(6, "RB2in M", 1);
    me_RPC_pass_1D->setBinLabel(7, "RB2in F", 1);
    me_RPC_pass_1D->setBinLabel(8, "RB2out B", 1);
    me_RPC_pass_1D->setBinLabel(9, "RB2out F", 1);
  } else {
    me_RPC_pass_1D->setBinLabel(5, "RB2in B", 1);
    me_RPC_pass_1D->setBinLabel(6, "RB2in F", 1);
    me_RPC_pass_1D->setBinLabel(7, "RB2out B", 1);
    me_RPC_pass_1D->setBinLabel(8, "RB2out M", 1);
    me_RPC_pass_1D->setBinLabel(9, "RB2out F", 1);
  }
  me_RPC_pass_1D->setBinLabel(10, "RB3- B", 1);
  me_RPC_pass_1D->setBinLabel(11, "RB3- F", 1);
  me_RPC_pass_1D->setBinLabel(12, "RB3+ B", 1);
  me_RPC_pass_1D->setBinLabel(13, "RB3+ F", 1);
  me_RPC_pass_1D->setBinLabel(14, "RB4- B", 1);
  me_RPC_pass_1D->setBinLabel(15, "RB4- F", 1);
  me_RPC_pass_1D->setBinLabel(16, "RB4+ B", 1);
  me_RPC_pass_1D->setBinLabel(17, "RB4+ F", 1);
  me_RPC_pass_1D->setBinLabel(18, "RB4-- B", 1);
  me_RPC_pass_1D->setBinLabel(19, "RB4-- F", 1);
  me_RPC_pass_1D->setBinLabel(20, "RB4++ B", 1);
  me_RPC_pass_1D->setBinLabel(21, "RB4++ F", 1);
  me_RPC_pass->setAxisTitle("Number of passing probes", 2);

  me_RPC_fail_1D->setBinLabel(1, "RB1in B", 1);
  me_RPC_fail_1D->setBinLabel(2, "RB1in F", 1);
  me_RPC_fail_1D->setBinLabel(3, "RB1out B", 1);
  me_RPC_fail_1D->setBinLabel(4, "RB1out F", 1);
  if (std::abs(wheel) < 2) {
    me_RPC_fail_1D->setBinLabel(5, "RB2in B", 1);
    me_RPC_fail_1D->setBinLabel(6, "RB2in M", 1);
    me_RPC_fail_1D->setBinLabel(7, "RB2in F", 1);
    me_RPC_fail_1D->setBinLabel(8, "RB2out B", 1);
    me_RPC_fail_1D->setBinLabel(9, "RB2out F", 1);
  } else {
    me_RPC_fail_1D->setBinLabel(5, "RB2in B", 1);
    me_RPC_fail_1D->setBinLabel(6, "RB2in F", 1);
    me_RPC_fail_1D->setBinLabel(7, "RB2out B", 1);
    me_RPC_fail_1D->setBinLabel(8, "RB2out M", 1);
    me_RPC_fail_1D->setBinLabel(9, "RB2out F", 1);
  }
  me_RPC_fail_1D->setBinLabel(10, "RB3- B", 1);
  me_RPC_fail_1D->setBinLabel(11, "RB3- F", 1);
  me_RPC_fail_1D->setBinLabel(12, "RB3+ B", 1);
  me_RPC_fail_1D->setBinLabel(13, "RB3+ F", 1);
  me_RPC_fail_1D->setBinLabel(14, "RB4- B", 1);
  me_RPC_fail_1D->setBinLabel(15, "RB4- F", 1);
  me_RPC_fail_1D->setBinLabel(16, "RB4+ B", 1);
  me_RPC_fail_1D->setBinLabel(17, "RB4+ F", 1);
  me_RPC_fail_1D->setBinLabel(18, "RB4-- B", 1);
  me_RPC_fail_1D->setBinLabel(19, "RB4-- F", 1);
  me_RPC_fail_1D->setBinLabel(20, "RB4++ B", 1);
  me_RPC_fail_1D->setBinLabel(21, "RB4++ F", 1);
  me_RPC_fail_1D->setAxisTitle("Number of failing probes", 2);

  m_histos[hName_RPC_pass] = me_RPC_pass;
  m_histos[hName_RPC_fail] = me_RPC_fail;

  m_histos[hName_RPC_pass_1D] = me_RPC_pass_1D;
  m_histos[hName_RPC_fail_1D] = me_RPC_fail_1D;
}

void RPCTnPEfficiencyTask::bookEndcapHistos(DQMStore::IBooker& iBooker, int station, std::string folder) {
  auto baseDir = topFolder() + folder + "/";
  iBooker.setCurrentFolder(baseDir);

  LogTrace("DQMOffline|MuonDPG|RPCTnPEfficiencyTask")
      << "[RPCTnPEfficiencyTask]: booking histos in " << baseDir << std::endl;

  auto hName_RPC_pass = std::string("RPC_nPassingProbePerRoll_Endcap_Sta") + std::to_string(station);
  auto hName_RPC_fail = std::string("RPC_nFailingProbePerRoll_Endcap_Sta") + std::to_string(station);

  auto hName_RPC_pass_1D = std::string("RPC_nPassingProbePerRoll_Endcap_1D_Sta") + std::to_string(station);
  auto hName_RPC_fail_1D = std::string("RPC_nFailingProbePerRoll_Endcap_1D_Sta") + std::to_string(station);

  MonitorElement* me_RPC_pass =
      iBooker.book2D(hName_RPC_pass.c_str(), hName_RPC_pass.c_str(), 36, 0.5, 36.5, 6, 0.5, 6.5);
  MonitorElement* me_RPC_fail =
      iBooker.book2D(hName_RPC_fail.c_str(), hName_RPC_fail.c_str(), 36, 0.5, 36.5, 6, 0.5, 6.5);

  MonitorElement* me_RPC_pass_1D = iBooker.book1D(hName_RPC_pass_1D.c_str(), hName_RPC_pass_1D.c_str(), 6, 0.5, 6.5);
  MonitorElement* me_RPC_fail_1D = iBooker.book1D(hName_RPC_fail_1D.c_str(), hName_RPC_fail_1D.c_str(), 6, 0.5, 6.5);

  me_RPC_pass->setBinLabel(1, "R2_A", 2);
  me_RPC_pass->setBinLabel(2, "R2_B", 2);
  me_RPC_pass->setBinLabel(3, "R2_C", 2);
  me_RPC_pass->setBinLabel(4, "R3_A", 2);
  me_RPC_pass->setBinLabel(5, "R3_B", 2);
  me_RPC_pass->setBinLabel(6, "R3_C", 2);
  for (int i = 1; i < 37; ++i) {
    me_RPC_pass->setBinLabel(i, std::to_string(i), 1);
  }
  me_RPC_pass->setAxisTitle("Sector", 1);
  me_RPC_pass->setAxisTitle("Number of passing probes", 3);
  me_RPC_pass->setTitle("RE" + std::to_string(station));

  me_RPC_fail->setBinLabel(1, "R2_A", 2);
  me_RPC_fail->setBinLabel(2, "R2_B", 2);
  me_RPC_fail->setBinLabel(3, "R2_C", 2);
  me_RPC_fail->setBinLabel(4, "R3_A", 2);
  me_RPC_fail->setBinLabel(5, "R3_B", 2);
  me_RPC_fail->setBinLabel(6, "R3_C", 2);
  for (int i = 1; i < 37; ++i) {
    me_RPC_fail->setBinLabel(i, std::to_string(i), 1);
  }
  me_RPC_fail->setAxisTitle("Sector", 1);
  me_RPC_fail->setAxisTitle("Number of failing probes", 3);
  me_RPC_fail->setTitle("RE" + std::to_string(station));

  me_RPC_pass_1D->setBinLabel(1, "R2_A", 1);
  me_RPC_pass_1D->setBinLabel(2, "R2_B", 1);
  me_RPC_pass_1D->setBinLabel(3, "R2_C", 1);
  me_RPC_pass_1D->setBinLabel(4, "R3_A", 1);
  me_RPC_pass_1D->setBinLabel(5, "R3_B", 1);
  me_RPC_pass_1D->setBinLabel(6, "R3_C", 1);
  me_RPC_pass_1D->setAxisTitle("Number of passing probes", 2);
  me_RPC_pass_1D->setTitle("RE" + std::to_string(station));

  me_RPC_fail_1D->setBinLabel(1, "R2_A", 1);
  me_RPC_fail_1D->setBinLabel(2, "R2_B", 1);
  me_RPC_fail_1D->setBinLabel(3, "R2_C", 1);
  me_RPC_fail_1D->setBinLabel(4, "R3_A", 1);
  me_RPC_fail_1D->setBinLabel(5, "R3_B", 1);
  me_RPC_fail_1D->setBinLabel(6, "R3_C", 1);
  me_RPC_fail_1D->setAxisTitle("Number of failing probes", 2);
  me_RPC_fail_1D->setTitle("RE" + std::to_string(station));

  m_histos[hName_RPC_pass] = me_RPC_pass;
  m_histos[hName_RPC_fail] = me_RPC_fail;

  m_histos[hName_RPC_pass_1D] = me_RPC_pass_1D;
  m_histos[hName_RPC_fail_1D] = me_RPC_fail_1D;
}

int RPCTnPEfficiencyTask::get_barrel_histo_ycoord(
    int ring, int station, int sector, int layer, int subsector, int roll) {
  int ycoord;

  if (station < 3) {
    //There are three rolls in RB2in for wheel=-1,0,+1 and in RB2out for wheel=-2,+2
    bool three_rolls = station == 2 && ((std::abs(ring) > 1 && layer == 2) || (std::abs(ring) < 2 && layer == 1));

    int layer_roll;

    if (!three_rolls) {
      roll = roll > 1 ? 2 : 1;
      int a = station == 2 && std::abs(ring) < 2 && layer == 2 ? 3 : 2;
      layer_roll = (a * (layer - 1)) + roll;
    } else {
      layer_roll = (2 * (layer - 1)) + roll;
    }

    ycoord = (4 * (station - 1)) + layer_roll;
  } else if (station == 3) {
    roll = roll > 1 ? 2 : 1;
    ycoord = 9 + (4 * (station - 3)) + (2 * (subsector - 1)) + roll;
  } else {
    int my_subsector = subsector;
    //The numbering scheme of subsector in sector 4
    //of station 4 does not match the bins order in the plot:
    //_____SUBSECTOR_____|_______BIN_ORDERING_____
    // ++ --> subsector 4| RB4++ --> my_subsector 4
    //  + --> subsector 3| RB4-- --> my_subsector 3
    //  - --> subsector 2| RB4+  --> my_subsector 2
    // -- --> subsector 1| RB4-  --> my_subsector 1

    if (sector == 4) {
      switch (subsector) {
        case 1:
          my_subsector = 3;
          break;
        case 2:
          my_subsector = 1;
          break;
        case 3:
          my_subsector = 2;
          break;
        case 4:
          my_subsector = 4;
          break;
      }
    }
    roll = roll > 1 ? 2 : 1;
    ycoord = 9 + (4 * (station - 3)) + (2 * (my_subsector - 1)) + roll;
  }

  return ycoord;
}

std::string RPCTnPEfficiencyTask::topFolder() const { return "RPC/Segment_TnP/"; };

DEFINE_FWK_MODULE(RPCTnPEfficiencyTask);
