/*
 * \file DTTnPEfficiencyTask.cc
 *
 * \author L. Lunerti - INFN Bologna
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/MuonSegmentMatch.h"

#include "DQMOffline/MuonDPG/interface/BaseTnPEfficiencyTask.h"

class DTTnPEfficiencyTask : public BaseTnPEfficiencyTask {
public:
  /// Constructor
  DTTnPEfficiencyTask(const edm::ParameterSet& config);

  /// Destructor
  ~DTTnPEfficiencyTask() override;

protected:
  std::string topFolder() const override;

  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& context) override;

  /// Book wheel granularity histograms
  void bookWheelHistos(DQMStore::IBooker& iBooker, int wheel, std::string folder = "");

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context) override;
};

DTTnPEfficiencyTask::DTTnPEfficiencyTask(const edm::ParameterSet& config) : BaseTnPEfficiencyTask(config) {
  LogTrace("DQMOffline|MuonDPG|DTTnPEfficiencyTask") << "[DTTnPEfficiencyTask]: Constructor" << std::endl;
}

DTTnPEfficiencyTask::~DTTnPEfficiencyTask() {
  LogTrace("DQMOffline|MuonDPG|DTTnPEfficiencyTask")
      << "[DTTnPEfficiencyTask]: analyzed " << m_nEvents << " events" << std::endl;
}

void DTTnPEfficiencyTask::bookHistograms(DQMStore::IBooker& iBooker,
                                         edm::Run const& run,
                                         edm::EventSetup const& context) {
  BaseTnPEfficiencyTask::bookHistograms(iBooker, run, context);

  LogTrace("DQMOffline|MuonDPG|DTTnPEfficiencyTask") << "[DTTnPEfficiencyTask]: bookHistograms" << std::endl;

  for (int wheel = -2; wheel <= 2; ++wheel) {
    bookWheelHistos(iBooker, wheel, "Task");
  }
  auto baseDir = topFolder() + "Task/";
  iBooker.setCurrentFolder(baseDir);

  MonitorElement* me_DT_pass_allCh = iBooker.book1D("DT_nPassingProbe_allCh", "DT_nPassingProbe_allCh", 20, 0.5, 20.5);
  MonitorElement* me_DT_fail_allCh = iBooker.book1D("DT_nFailingProbe_allCh", "DT_nFailingProbe_allCh", 20, 0.5, 20.5);

  me_DT_pass_allCh->setBinLabel(1, "MB1/YB-2", 1);
  me_DT_pass_allCh->setBinLabel(2, "MB2/YB-2", 1);
  me_DT_pass_allCh->setBinLabel(3, "MB3/YB-2", 1);
  me_DT_pass_allCh->setBinLabel(4, "MB4/YB-2", 1);
  me_DT_pass_allCh->setBinLabel(5, "MB1/YB-1", 1);
  me_DT_pass_allCh->setBinLabel(6, "MB2/YB-1", 1);
  me_DT_pass_allCh->setBinLabel(7, "MB3/YB-1", 1);
  me_DT_pass_allCh->setBinLabel(8, "MB4/YB-1", 1);
  me_DT_pass_allCh->setBinLabel(9, "MB1/YB0", 1);
  me_DT_pass_allCh->setBinLabel(10, "MB2/YB0", 1);
  me_DT_pass_allCh->setBinLabel(11, "MB3/YB0", 1);
  me_DT_pass_allCh->setBinLabel(12, "MB4/YB0", 1);
  me_DT_pass_allCh->setBinLabel(13, "MB1/YB1", 1);
  me_DT_pass_allCh->setBinLabel(14, "MB2/YB1", 1);
  me_DT_pass_allCh->setBinLabel(15, "MB3/YB1", 1);
  me_DT_pass_allCh->setBinLabel(16, "MB4/YB1", 1);
  me_DT_pass_allCh->setBinLabel(17, "MB1/YB2", 1);
  me_DT_pass_allCh->setBinLabel(18, "MB2/YB2", 1);
  me_DT_pass_allCh->setBinLabel(19, "MB3/YB2", 1);
  me_DT_pass_allCh->setBinLabel(20, "MB4/YB2", 1);
  me_DT_pass_allCh->setAxisTitle("Number of passing probes", 2);

  me_DT_fail_allCh->setBinLabel(1, "MB1/YB-2", 1);
  me_DT_fail_allCh->setBinLabel(2, "MB2/YB-2", 1);
  me_DT_fail_allCh->setBinLabel(3, "MB3/YB-2", 1);
  me_DT_fail_allCh->setBinLabel(4, "MB4/YB-2", 1);
  me_DT_fail_allCh->setBinLabel(5, "MB1/YB-1", 1);
  me_DT_fail_allCh->setBinLabel(6, "MB2/YB-1", 1);
  me_DT_fail_allCh->setBinLabel(7, "MB3/YB-1", 1);
  me_DT_fail_allCh->setBinLabel(8, "MB4/YB-1", 1);
  me_DT_fail_allCh->setBinLabel(9, "MB1/YB0", 1);
  me_DT_fail_allCh->setBinLabel(10, "MB2/YB0", 1);
  me_DT_fail_allCh->setBinLabel(11, "MB3/YB0", 1);
  me_DT_fail_allCh->setBinLabel(12, "MB4/YB0", 1);
  me_DT_fail_allCh->setBinLabel(13, "MB1/YB1", 1);
  me_DT_fail_allCh->setBinLabel(14, "MB2/YB1", 1);
  me_DT_fail_allCh->setBinLabel(15, "MB3/YB1", 1);
  me_DT_fail_allCh->setBinLabel(16, "MB4/YB1", 1);
  me_DT_fail_allCh->setBinLabel(17, "MB1/YB2", 1);
  me_DT_fail_allCh->setBinLabel(18, "MB2/YB2", 1);
  me_DT_fail_allCh->setBinLabel(19, "MB3/YB2", 1);
  me_DT_fail_allCh->setBinLabel(20, "MB4/YB2", 1);
  me_DT_fail_allCh->setAxisTitle("Number of failing probes", 2);

  m_histos["DT_nPassingProbe_allCh"] = me_DT_pass_allCh;
  m_histos["DT_nFailingProbe_allCh"] = me_DT_fail_allCh;
}

void DTTnPEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& context) {
  BaseTnPEfficiencyTask::analyze(event, context);

  edm::Handle<reco::MuonCollection> muons;
  event.getByToken(m_muToken, muons);

  //DT variables
  std::vector<std::vector<int>> probe_coll_DT_wh;
  std::vector<std::vector<int>> probe_coll_DT_sec;
  std::vector<std::vector<int>> probe_coll_DT_sta;
  std::vector<std::vector<float>> probe_coll_DT_dx;
  std::vector<uint8_t> probe_coll_DT_staMatch;

  std::vector<unsigned> probe_indices;
  if (!m_probeIndices.empty())
    probe_indices = m_probeIndices.back();

  //Fill probe dx + subdetector coordinates
  for (const auto i : probe_indices) {
    //DT variables
    std::vector<int> probe_DT_wh;
    std::vector<int> probe_DT_sec;
    std::vector<int> probe_DT_sta;
    std::vector<float> probe_DT_dx;
    uint8_t DT_stationMatching = 0;

    float dt_matched = false;  // fill detailed plots only for probes matching DT

    for (const auto& chambMatch : (*muons).at(i).matches()) {
      // look in DTs
      if (chambMatch.detector() == MuonSubdetId::DT) {
        if (chambMatch.edgeX < m_borderCut && chambMatch.edgeY < m_borderCut) {
          dt_matched = true;  //fill detailed plots if at least one CSC match

          DTChamberId chId(chambMatch.id.rawId());

          int wheel = chId.wheel();
          int sector = chId.sector();
          int station = chId.station();

          reco::MuonSegmentMatch closest_matchedSegment;
          double smallestDx = 999.;

          for (auto& seg : chambMatch.segmentMatches) {
            float dx = std::abs(chambMatch.x - seg.x);
            if (dx < smallestDx) {
              smallestDx = dx;
              closest_matchedSegment = seg;
            }
          }

          DT_stationMatching = DT_stationMatching | (1 << (station - 1));

          probe_DT_wh.push_back(wheel);
          probe_DT_sec.push_back(sector);
          probe_DT_sta.push_back(station);
          probe_DT_dx.push_back(smallestDx);
        }
      } else
        continue;
    }  //loop over chamber matches

    //Fill detailed plots
    if (m_detailedAnalysis && dt_matched) {
      m_histos.find("probeEta")->second->Fill((*muons).at(i).eta());
      m_histos.find("probePhi")->second->Fill((*muons).at(i).phi());
      m_histos.find("probeNumberOfMatchedStations")->second->Fill((*muons).at(i).numberOfMatchedStations());
      m_histos.find("probePt")->second->Fill((*muons).at(i).pt());
    }

    //Fill DT variables
    probe_coll_DT_wh.push_back(probe_DT_wh);
    probe_coll_DT_sec.push_back(probe_DT_sec);
    probe_coll_DT_sta.push_back(probe_DT_sta);
    probe_coll_DT_dx.push_back(probe_DT_dx);
    probe_coll_DT_staMatch.push_back(DT_stationMatching);
  }  //loop over probe collection

  //Loop over probes
  for (unsigned i = 0; i < probe_indices.size(); ++i) {
    uint8_t DT_matchPatt = probe_coll_DT_staMatch.at(i);

    //Loop over DT matches
    unsigned nDT_matches = probe_coll_DT_wh.at(i).size();
    for (unsigned j = 0; j < nDT_matches; ++j) {
      //DT variables
      int DT_wheel = probe_coll_DT_wh.at(i).at(j);
      int DT_station = probe_coll_DT_sta.at(i).at(j);
      int DT_sector = probe_coll_DT_sec.at(i).at(j);
      float DT_dx = probe_coll_DT_dx.at(i).at(j);

      //Fill DT plots
      if ((DT_matchPatt & (1 << (DT_station - 1))) != 0 &&  //avoids 0 station matching
          (DT_matchPatt & (1 << (DT_station - 1))) !=
              DT_matchPatt)  //avoids matching with the station under consideration only
      {
        if (DT_dx < m_dxCut) {
          std::string hName = std::string("DT_nPassingProbePerCh_W") + std::to_string(DT_wheel);
          m_histos.find(hName)->second->Fill(DT_sector, DT_station);
          m_histos.find("DT_nPassingProbe_allCh")->second->Fill((DT_station) + 4 * (DT_wheel + 2));
        } else {
          std::string hName = std::string("DT_nFailingProbePerCh_W") + std::to_string(DT_wheel);
          m_histos.find(hName)->second->Fill(DT_sector, DT_station);
          m_histos.find("DT_nFailingProbe_allCh")->second->Fill((DT_station) + 4 * (DT_wheel + 2));
        }
      }
    }
  }
}

void DTTnPEfficiencyTask::bookWheelHistos(DQMStore::IBooker& iBooker, int wheel, std::string folder) {
  auto baseDir = topFolder() + folder + "/";
  iBooker.setCurrentFolder(baseDir);

  LogTrace("DQMOffline|MuonDPG|DTTnPEfficiencyTask")
      << "[DTTnPEfficiencyTask]: booking histos in " << baseDir << std::endl;

  auto hName_DT_pass = std::string("DT_nPassingProbePerCh_W") + std::to_string(wheel);
  auto hName_DT_fail = std::string("DT_nFailingProbePerCh_W") + std::to_string(wheel);

  MonitorElement* me_DT_pass = iBooker.book2D(hName_DT_pass.c_str(), hName_DT_pass.c_str(), 14, 0.5, 14.5, 4, 0., 4.5);
  MonitorElement* me_DT_fail = iBooker.book2D(hName_DT_fail.c_str(), hName_DT_fail.c_str(), 14, 0.5, 14.5, 4, 0., 4.5);

  me_DT_pass->setBinLabel(1, "MB1", 2);
  me_DT_pass->setBinLabel(2, "MB2", 2);
  me_DT_pass->setBinLabel(3, "MB3", 2);
  me_DT_pass->setBinLabel(4, "MB4", 2);
  for (int i = 1; i < 15; ++i) {
    me_DT_pass->setBinLabel(i, std::to_string(i), 1);
  }
  me_DT_pass->setAxisTitle("Sector", 1);
  me_DT_pass->setAxisTitle("Number of passing probes", 3);

  me_DT_fail->setBinLabel(1, "MB1", 2);
  me_DT_fail->setBinLabel(2, "MB2", 2);
  me_DT_fail->setBinLabel(3, "MB3", 2);
  me_DT_fail->setBinLabel(4, "MB4", 2);
  for (int i = 1; i < 15; ++i) {
    me_DT_fail->setBinLabel(i, std::to_string(i), 1);
  }
  me_DT_fail->setAxisTitle("Sector", 1);
  me_DT_fail->setAxisTitle("Number of failing probes", 3);

  m_histos[hName_DT_pass] = me_DT_pass;
  m_histos[hName_DT_fail] = me_DT_fail;
}

std::string DTTnPEfficiencyTask::topFolder() const { return "DT/Segment_TnP/"; };

DEFINE_FWK_MODULE(DTTnPEfficiencyTask);
