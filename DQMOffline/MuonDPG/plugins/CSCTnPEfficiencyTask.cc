/*
 * \file CSCTnPEfficiencyTask.cc
 *
 * \author L. Lunerti - INFN Bologna
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/MuonSegmentMatch.h"

#include "DQMOffline/MuonDPG/interface/BaseTnPEfficiencyTask.h"

class CSCTnPEfficiencyTask : public BaseTnPEfficiencyTask {
public:
  /// Constructor
  CSCTnPEfficiencyTask(const edm::ParameterSet& config);

  /// Destructor
  ~CSCTnPEfficiencyTask() override;

protected:
  std::string topFolder() const override;

  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& context) override;

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context) override;
};

CSCTnPEfficiencyTask::CSCTnPEfficiencyTask(const edm::ParameterSet& config) : BaseTnPEfficiencyTask(config) {
  LogTrace("DQMOffline|MuonDPG|CSCTnPEfficiencyTask") << "[CSCTnPEfficiencyTask]: Constructor" << std::endl;
}

CSCTnPEfficiencyTask::~CSCTnPEfficiencyTask() {
  LogTrace("DQMOffline|MuonDPG|CSCTnPEfficiencyTask")
      << "[CSCTnPEfficiencyTask]: analyzed " << m_nEvents << " events" << std::endl;
}

void CSCTnPEfficiencyTask::bookHistograms(DQMStore::IBooker& iBooker,
                                          edm::Run const& run,
                                          edm::EventSetup const& context) {
  BaseTnPEfficiencyTask::bookHistograms(iBooker, run, context);

  LogTrace("DQMOffline|MuonDPG|CSCTnPEfficiencyTask") << "[CSCTnPEfficiencyTask]: bookHistograms" << std::endl;

  auto baseDir = topFolder() + "Task/";
  iBooker.setCurrentFolder(baseDir);

  MonitorElement* me_CSC_pass_allCh =
      iBooker.book2D("CSC_nPassingProbe_allCh", "CSC_nPassingProbe_allCh", 9, -4., 5., 4, 0., 4.5);
  MonitorElement* me_CSC_fail_allCh =
      iBooker.book2D("CSC_nFailingProbe_allCh", "CSC_nFailingProbe_allCh", 9, -4., 5., 4, 0., 4.5);

  MonitorElement* me_CSC_pass_allCh_1D =
      iBooker.book1D("CSC_nPassingProbe_allCh_1D", "CSC_nPassingProbe_allCh_1D", 9, -4., 5.);
  MonitorElement* me_CSC_fail_allCh_1D =
      iBooker.book1D("CSC_nFailingProbe_allCh_1D", "CSC_nFailingProbe_allCh_1D", 9, -4., 5.);

  me_CSC_pass_allCh->setBinLabel(1, "ME-4", 1);
  me_CSC_pass_allCh->setBinLabel(2, "ME-3", 1);
  me_CSC_pass_allCh->setBinLabel(3, "ME-2", 1);
  me_CSC_pass_allCh->setBinLabel(4, "ME-1", 1);
  me_CSC_pass_allCh->setBinLabel(6, "ME1", 1);
  me_CSC_pass_allCh->setBinLabel(7, "ME2", 1);
  me_CSC_pass_allCh->setBinLabel(8, "ME3", 1);
  me_CSC_pass_allCh->setBinLabel(9, "ME4", 1);
  for (int i = 1; i < 5; ++i) {
    me_CSC_pass_allCh->setBinLabel(i, std::to_string(i), 2);
  }
  me_CSC_pass_allCh->setAxisTitle("Ring", 2);
  me_CSC_pass_allCh->setAxisTitle("Number of passing probes", 3);

  me_CSC_fail_allCh->setBinLabel(1, "ME-4", 1);
  me_CSC_fail_allCh->setBinLabel(2, "ME-3", 1);
  me_CSC_fail_allCh->setBinLabel(3, "ME-2", 1);
  me_CSC_fail_allCh->setBinLabel(4, "ME-1", 1);
  me_CSC_fail_allCh->setBinLabel(6, "ME1", 1);
  me_CSC_fail_allCh->setBinLabel(7, "ME2", 1);
  me_CSC_fail_allCh->setBinLabel(8, "ME3", 1);
  me_CSC_fail_allCh->setBinLabel(9, "ME4", 1);
  for (int i = 1; i < 5; ++i) {
    me_CSC_fail_allCh->setBinLabel(i, std::to_string(i), 2);
  }
  me_CSC_fail_allCh->setAxisTitle("Ring", 2);
  me_CSC_fail_allCh->setAxisTitle("Number of failing probes", 3);

  me_CSC_pass_allCh_1D->setBinLabel(1, "ME-4", 1);
  me_CSC_pass_allCh_1D->setBinLabel(2, "ME-3", 1);
  me_CSC_pass_allCh_1D->setBinLabel(3, "ME-2", 1);
  me_CSC_pass_allCh_1D->setBinLabel(4, "ME-1", 1);
  me_CSC_pass_allCh_1D->setBinLabel(6, "ME1", 1);
  me_CSC_pass_allCh_1D->setBinLabel(7, "ME2", 1);
  me_CSC_pass_allCh_1D->setBinLabel(8, "ME3", 1);
  me_CSC_pass_allCh_1D->setBinLabel(9, "ME4", 1);
  me_CSC_pass_allCh_1D->setAxisTitle("Number of passing probes", 2);

  me_CSC_fail_allCh_1D->setBinLabel(1, "ME-4", 1);
  me_CSC_fail_allCh_1D->setBinLabel(2, "ME-3", 1);
  me_CSC_fail_allCh_1D->setBinLabel(3, "ME-2", 1);
  me_CSC_fail_allCh_1D->setBinLabel(4, "ME-1", 1);
  me_CSC_fail_allCh_1D->setBinLabel(6, "ME1", 1);
  me_CSC_fail_allCh_1D->setBinLabel(7, "ME2", 1);
  me_CSC_fail_allCh_1D->setBinLabel(8, "ME3", 1);
  me_CSC_fail_allCh_1D->setBinLabel(9, "ME4", 1);
  me_CSC_fail_allCh_1D->setAxisTitle("Number of failing probes", 2);

  m_histos["CSC_nPassingProbe_allCh"] = me_CSC_pass_allCh;
  m_histos["CSC_nFailingProbe_allCh"] = me_CSC_fail_allCh;

  m_histos["CSC_nPassingProbe_allCh_1D"] = me_CSC_pass_allCh_1D;
  m_histos["CSC_nFailingProbe_allCh_1D"] = me_CSC_fail_allCh_1D;
}

void CSCTnPEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& context) {
  BaseTnPEfficiencyTask::analyze(event, context);

  edm::Handle<reco::MuonCollection> muons;
  event.getByToken(m_muToken, muons);

  //CSC variables
  std::vector<std::vector<int>> probe_coll_CSC_zend;
  std::vector<std::vector<int>> probe_coll_CSC_ring;
  std::vector<std::vector<int>> probe_coll_CSC_sta;
  std::vector<std::vector<float>> probe_coll_CSC_dx;
  std::vector<uint8_t> probe_coll_CSC_staMatch;

  std::vector<unsigned> probe_indices;
  if (!m_probeIndices.empty())
    probe_indices = m_probeIndices.back();

  //Fill probe dx + subdetector coordinates
  for (const auto i : probe_indices) {
    //CSC variables
    std::vector<int> probe_CSC_zend;
    std::vector<int> probe_CSC_ring;
    std::vector<int> probe_CSC_sta;
    std::vector<float> probe_CSC_dx;
    uint8_t CSC_stationMatching = 0;

    float csc_matched = false;  // fill detailed plots only for probes matching CSC

    for (const auto& chambMatch : (*muons).at(i).matches()) {
      // look in CSCs
      if (chambMatch.detector() == MuonSubdetId::CSC) {
        if (chambMatch.edgeX < m_borderCut && chambMatch.edgeY < m_borderCut) {
          csc_matched = true;  //fill detailed plots if at least one CSC match

          CSCDetId chId(chambMatch.id.rawId());

          int zendcap = chId.zendcap();
          int ring = chId.ring();
          int station = chId.station();

          reco::MuonSegmentMatch closest_matchedSegment;
          double smallestDx = 99999.;
          for (auto& seg : chambMatch.segmentMatches) {
            float dx = std::abs(chambMatch.x - seg.x);
            if (dx < smallestDx) {
              smallestDx = dx;
              closest_matchedSegment = seg;
            }
          }

          CSC_stationMatching = CSC_stationMatching | (1 << (station - 1));

          if (station == 1 && ring == 4 && chambMatch.y < -31.5) {
            probe_CSC_zend.push_back(zendcap);
            probe_CSC_ring.push_back(ring);
            probe_CSC_sta.push_back(station);
            probe_CSC_dx.push_back(smallestDx);
          } else if (station == 1 && ring == 1 && chambMatch.y > -31.5) {
            probe_CSC_zend.push_back(zendcap);
            probe_CSC_ring.push_back(ring);
            probe_CSC_sta.push_back(station);
            probe_CSC_dx.push_back(smallestDx);
          } else if (station > 1 || ring == 2 || ring == 3) {
            probe_CSC_zend.push_back(zendcap);
            probe_CSC_ring.push_back(ring);
            probe_CSC_sta.push_back(station);
            probe_CSC_dx.push_back(smallestDx);
          }
        }
      } else
        continue;
    }  //loop over chamber matches

    //Fill detailed plots
    if (m_detailedAnalysis && csc_matched) {
      m_histos.find("probeEta")->second->Fill((*muons).at(i).eta());
      m_histos.find("probePhi")->second->Fill((*muons).at(i).phi());
      m_histos.find("probeNumberOfMatchedStations")->second->Fill((*muons).at(i).numberOfMatchedStations());
      m_histos.find("probePt")->second->Fill((*muons).at(i).pt());
    }

    //Fill CSC variables
    probe_coll_CSC_zend.push_back(probe_CSC_zend);
    probe_coll_CSC_ring.push_back(probe_CSC_ring);
    probe_coll_CSC_sta.push_back(probe_CSC_sta);
    probe_coll_CSC_dx.push_back(probe_CSC_dx);
    probe_coll_CSC_staMatch.push_back(CSC_stationMatching);
  }  //loop over probe collection

  //Loop over probes
  for (unsigned i = 0; i < probe_indices.size(); ++i) {
    uint8_t CSC_matchPatt = probe_coll_CSC_staMatch.at(i);

    //Loop over CSC matches
    unsigned nCSC_matches = probe_coll_CSC_zend.at(i).size();
    for (unsigned j = 0; j < nCSC_matches; ++j) {
      int CSC_zendcap = probe_coll_CSC_zend.at(i).at(j);
      int CSC_sta = probe_coll_CSC_sta.at(i).at(j);
      int CSC_ring = probe_coll_CSC_ring.at(i).at(j);
      float CSC_dx = probe_coll_CSC_dx.at(i).at(j);

      //Fill CSC plots
      if ((CSC_matchPatt & (1 << (CSC_sta - 1))) != 0 &&  //avoids 0 station matching
          (CSC_matchPatt & (1 << (CSC_sta - 1))) !=
              CSC_matchPatt)  //avoids matching with the station under consideration only
      {
        if (CSC_dx < m_dxCut) {
          m_histos.find("CSC_nPassingProbe_allCh")->second->Fill(CSC_zendcap * CSC_sta, CSC_ring);
          m_histos.find("CSC_nPassingProbe_allCh_1D")->second->Fill(CSC_zendcap * CSC_sta);
        } else {
          m_histos.find("CSC_nFailingProbe_allCh")->second->Fill(CSC_zendcap * CSC_sta, CSC_ring);
          m_histos.find("CSC_nFailingProbe_allCh_1D")->second->Fill(CSC_zendcap * CSC_sta);
        }
      }
    }
  }
}

std::string CSCTnPEfficiencyTask::topFolder() const { return "CSC/Segment_TnP/"; };

DEFINE_FWK_MODULE(CSCTnPEfficiencyTask);
