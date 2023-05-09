/*
 * \file GEMTnPEfficiencyTask.cc
 * \author Qianying
 *
 * \interited from the TnP framework of  
 * \author L. Lunerti - INFN Bologna
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/MuonSegmentMatch.h"
#include "DataFormats/MuonReco/interface/MuonGEMHitMatch.h"

#include "DQMOffline/MuonDPG/interface/BaseTnPEfficiencyTask.h"

class GEMTnPEfficiencyTask : public BaseTnPEfficiencyTask {
public:
  /// Constructor
  GEMTnPEfficiencyTask(const edm::ParameterSet& config);

  /// Destructor
  ~GEMTnPEfficiencyTask() override;

protected:
  std::string topFolder() const override;

  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& context) override;

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context) override;
};

GEMTnPEfficiencyTask::GEMTnPEfficiencyTask(const edm::ParameterSet& config) : BaseTnPEfficiencyTask(config) {
  LogTrace("DQMOffline|MuonDPG|GEMTnPEfficiencyTask") << "[GEMTnPEfficiencyTask]: Constructor" << std::endl;
}

GEMTnPEfficiencyTask::~GEMTnPEfficiencyTask() {
  LogTrace("DQMOffline|MuonDPG|GEMTnPEfficiencyTask")
      << "[GEMTnPEfficiencyTask]: analyzed " << m_nEvents << " events" << std::endl;
}

void GEMTnPEfficiencyTask::bookHistograms(DQMStore::IBooker& iBooker,
                                          edm::Run const& run,
                                          edm::EventSetup const& context) {
  BaseTnPEfficiencyTask::bookHistograms(iBooker, run, context);

  LogTrace("DQMOffline|MuonDPG|GEMTnPEfficiencyTask") << "[GEMTnPEfficiencyTask]: bookHistograms" << std::endl;

  auto baseDir = topFolder() + "Task/";
  iBooker.setCurrentFolder(baseDir);

  MonitorElement* me_GEM_pass_Ch_region =
      iBooker.book2D("GEM_nPassingProbe_Ch_region", "GEM_nPassingProbe_Ch_region", 2, -1.5, 1.5, 36, 1, 37);
  MonitorElement* me_GEM_fail_Ch_region =
      iBooker.book2D("GEM_nFailingProbe_Ch_region", "GEM_nFailingProbe_Ch_region", 2, -1.5, 1.5, 36, 1, 37);
  MonitorElement* me_GEM_pass_Ch_region_GE1 =
      iBooker.book2D("GEM_nPassingProbe_Ch_region_GE1", "GEM_nPassingProbe_Ch_region_GE1", 4, 0, 4, 36, 1, 37);
  MonitorElement* me_GEM_fail_Ch_region_GE1 =
      iBooker.book2D("GEM_nFailingProbe_Ch_region_GE1", "GEM_nFailingProbe_Ch_region_GE1", 4, 0, 4, 36, 1, 37);
  MonitorElement* me_GEM_pass_Ch_region_GE1_NoL =
      iBooker.book2D("GEM_nPassingProbe_Ch_region_GE1_NoL", "GEM_nPassingProbe_Ch_region_GE1_NoL", 2, 0, 2, 36, 1, 37);
  MonitorElement* me_GEM_fail_Ch_region_GE1_NoL =
      iBooker.book2D("GEM_nFailingProbe_Ch_region_GE1_NoL", "GEM_nFailingProbe_Ch_region_GE1_NoL", 2, 0, 2, 36, 1, 37);
  MonitorElement* me_GEM_pass_Ch_eta =
      iBooker.book2D("GEM_nPassingProbe_Ch_eta", "GEM_nPassingProbe_Ch_eta", 24, 0, 2.4, 36, 1, 37);
  MonitorElement* me_GEM_fail_Ch_eta =
      iBooker.book2D("GEM_nFailingProbe_Ch_eta", "GEM_nFailingProbe_Ch_eta", 24, 0, 2.4, 36, 1, 37);
  MonitorElement* me_GEM_pass_Ch_phi =
      iBooker.book2D("GEM_nPassingProbe_Ch_phi", "GEM_nPassingProbe_Ch_phi", 20, -TMath::Pi(), TMath::Pi(), 36, 1, 37);
  MonitorElement* me_GEM_fail_Ch_phi =
      iBooker.book2D("GEM_nFailingProbe_Ch_phi", "GEM_nFailingProbe_Ch_phi", 20, -TMath::Pi(), TMath::Pi(), 36, 1, 37);
  MonitorElement* me_GEM_pass_allCh_1D =
      iBooker.book1D("GEM_nPassingProbe_allCh_1D", "GEM_nPassingProbe_allCh_1D", 2, -1.5, 1.5);
  MonitorElement* me_GEM_fail_allCh_1D =
      iBooker.book1D("GEM_nFailingProbe_allCh_1D", "GEM_nFailingProbe_allCh_1D", 2, -1.5, 1.5);
  MonitorElement* me_GEM_pass_chamber_1D =
      iBooker.book1D("GEM_nPassingProbe_chamber_1D", "GEM_nPassingProbe_chamber_1D", 36, 1, 37);
  MonitorElement* me_GEM_fail_chamber_1D =
      iBooker.book1D("GEM_nFailingProbe_chamber_1D", "GEM_nFailingProbe_chamber_1D", 36, 1, 37);
  MonitorElement* me_GEM_pass_chamber_p1_1D =
      iBooker.book1D("GEM_nPassingProbe_chamber_p1_1D", "GEM_nPassingProbe_chamber_p1_1D", 36, 1, 37);
  MonitorElement* me_GEM_fail_chamber_p1_1D =
      iBooker.book1D("GEM_nFailingProbe_chamber_p1_1D", "GEM_nFailingProbe_chamber_p1_1D", 36, 1, 37);
  MonitorElement* me_GEM_pass_chamber_p2_1D =
      iBooker.book1D("GEM_nPassingProbe_chamber_p2_1D", "GEM_nPassingProbe_chamber_p2_1D", 36, 1, 37);
  MonitorElement* me_GEM_fail_chamber_p2_1D =
      iBooker.book1D("GEM_nFailingProbe_chamber_p2_1D", "GEM_nFailingProbe_chamber_p2_1D", 36, 1, 37);
  MonitorElement* me_GEM_pass_chamber_n1_1D =
      iBooker.book1D("GEM_nPassingProbe_chamber_n1_1D", "GEM_nPassingProbe_chamber_n1_1D", 36, 1, 37);
  MonitorElement* me_GEM_fail_chamber_n1_1D =
      iBooker.book1D("GEM_nFailingProbe_chamber_n1_1D", "GEM_nFailingProbe_chamber_n1_1D", 36, 1, 37);
  MonitorElement* me_GEM_pass_chamber_n2_1D =
      iBooker.book1D("GEM_nPassingProbe_chamber_n2_1D", "GEM_nPassingProbe_chamber_n2_1D", 36, 1, 37);
  MonitorElement* me_GEM_fail_chamber_n2_1D =
      iBooker.book1D("GEM_nFailingProbe_chamber_n2_1D", "GEM_nFailingProbe_chamber_n2_1D", 36, 1, 37);
  //
  MonitorElement* me_GEM_pass_pt_1D = iBooker.book1D("GEM_nPassingProbe_pt_1D", "GEM_nPassingProbe_pt_1D", 20, 0, 100);
  MonitorElement* me_GEM_fail_pt_1D = iBooker.book1D("GEM_nFailingProbe_pt_1D", "GEM_nFailingProbe_pt_1D", 20, 0, 100);
  MonitorElement* me_GEM_pass_eta_1D =
      iBooker.book1D("GEM_nPassingProbe_eta_1D", "GEM_nPassingProbe_eta_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_fail_eta_1D =
      iBooker.book1D("GEM_nFailingProbe_eta_1D", "GEM_nFailingProbe_eta_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_pass_phi_1D =
      iBooker.book1D("GEM_nPassingProbe_phi_1D", "GEM_nPassingProbe_phi_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_fail_phi_1D =
      iBooker.book1D("GEM_nFailingProbe_phi_1D", "GEM_nFailingProbe_phi_1D", 20, -TMath::Pi(), TMath::Pi());
  ///
  MonitorElement* me_GEM_pass_pt_p1_1D =
      iBooker.book1D("GEM_nPassingProbe_pt_p1_1D", "GEM_nPassingProbe_pt_p1_1D", 20, 0, 100);
  MonitorElement* me_GEM_fail_pt_p1_1D =
      iBooker.book1D("GEM_nFailingProbe_pt_p1_1D", "GEM_nFailingProbe_pt_p1_1D", 20, 0, 100);
  MonitorElement* me_GEM_pass_eta_p1_1D =
      iBooker.book1D("GEM_nPassingProbe_eta_p1_1D", "GEM_nPassingProbe_eta_p1_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_fail_eta_p1_1D =
      iBooker.book1D("GEM_nFailingProbe_eta_p1_1D", "GEM_nFailingProbe_eta_p1_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_pass_phi_p1_1D =
      iBooker.book1D("GEM_nPassingProbe_phi_p1_1D", "GEM_nPassingProbe_phi_p1_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_fail_phi_p1_1D =
      iBooker.book1D("GEM_nFailingProbe_phi_p1_1D", "GEM_nFailingProbe_phi_p1_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_pass_pt_p2_1D =
      iBooker.book1D("GEM_nPassingProbe_pt_p2_1D", "GEM_nPassingProbe_pt_p2_1D", 20, 0, 100);
  MonitorElement* me_GEM_fail_pt_p2_1D =
      iBooker.book1D("GEM_nFailingProbe_pt_p2_1D", "GEM_nFailingProbe_pt_p2_1D", 20, 0, 100);
  MonitorElement* me_GEM_pass_eta_p2_1D =
      iBooker.book1D("GEM_nPassingProbe_eta_p2_1D", "GEM_nPassingProbe_eta_p2_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_fail_eta_p2_1D =
      iBooker.book1D("GEM_nFailingProbe_eta_p2_1D", "GEM_nFailingProbe_eta_p2_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_pass_phi_p2_1D =
      iBooker.book1D("GEM_nPassingProbe_phi_p2_1D", "GEM_nPassingProbe_phi_p2_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_fail_phi_p2_1D =
      iBooker.book1D("GEM_nFailingProbe_phi_p2_1D", "GEM_nFailingProbe_phi_p2_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_pass_pt_n1_1D =
      iBooker.book1D("GEM_nPassingProbe_pt_n1_1D", "GEM_nPassingProbe_pt_n1_1D", 20, 0, 100);
  MonitorElement* me_GEM_fail_pt_n1_1D =
      iBooker.book1D("GEM_nFailingProbe_pt_n1_1D", "GEM_nFailingProbe_pt_n1_1D", 20, 0, 100);
  MonitorElement* me_GEM_pass_eta_n1_1D =
      iBooker.book1D("GEM_nPassingProbe_eta_n1_1D", "GEM_nPassingProbe_eta_n1_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_fail_eta_n1_1D =
      iBooker.book1D("GEM_nFailingProbe_eta_n1_1D", "GEM_nFailingProbe_eta_n1_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_pass_phi_n1_1D =
      iBooker.book1D("GEM_nPassingProbe_phi_n1_1D", "GEM_nPassingProbe_phi_n1_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_fail_phi_n1_1D =
      iBooker.book1D("GEM_nFailingProbe_phi_n1_1D", "GEM_nFailingProbe_phi_n1_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_pass_pt_n2_1D =
      iBooker.book1D("GEM_nPassingProbe_pt_n2_1D", "GEM_nPassingProbe_pt_n2_1D", 20, 0, 100);
  MonitorElement* me_GEM_fail_pt_n2_1D =
      iBooker.book1D("GEM_nFailingProbe_pt_n2_1D", "GEM_nFailingProbe_pt_n2_1D", 20, 0, 100);
  MonitorElement* me_GEM_pass_eta_n2_1D =
      iBooker.book1D("GEM_nPassingProbe_eta_n2_1D", "GEM_nPassingProbe_eta_n2_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_fail_eta_n2_1D =
      iBooker.book1D("GEM_nFailingProbe_eta_n2_1D", "GEM_nFailingProbe_eta_n2_1D", 24, 0, 2.4);
  MonitorElement* me_GEM_pass_phi_n2_1D =
      iBooker.book1D("GEM_nPassingProbe_phi_n2_1D", "GEM_nPassingProbe_phi_n2_1D", 20, -TMath::Pi(), TMath::Pi());
  MonitorElement* me_GEM_fail_phi_n2_1D =
      iBooker.book1D("GEM_nFailingProbe_phi_n2_1D", "GEM_nFailingProbe_phi_n2_1D", 20, -TMath::Pi(), TMath::Pi());
  ////
  MonitorElement* me_ME0_pass_chamber_1D =
      iBooker.book1D("ME0_nPassingProbe_chamber_1D", "ME0_nPassingProbe_chamber_1D", 18, 1, 19);
  MonitorElement* me_ME0_fail_chamber_1D =
      iBooker.book1D("ME0_nFailingProbe_chamber_1D", "ME0_nFailingProbe_chamber_1D", 18, 1, 19);
  MonitorElement* me_GEM_pass_Ch_region_layer_phase2 = iBooker.book2D(
      "GEM_nPassingProbe_Ch_region_layer_phase2", "GEM_nPassingProbe_Ch_region_layer_phase2", 10, 0, 10, 36, 1, 37);
  MonitorElement* me_GEM_fail_Ch_region_layer_phase2 = iBooker.book2D(
      "GEM_nFailingProbe_Ch_region_layer_phase2", "GEM_nFailingProbe_Ch_region_layer_phase2", 10, 0, 10, 36, 1, 37);

  me_GEM_pass_allCh_1D->setBinLabel(1, "GE-11", 1);
  me_GEM_pass_allCh_1D->setBinLabel(2, "GE11", 1);
  me_GEM_pass_allCh_1D->setAxisTitle("Number of passing probes", 2);

  me_GEM_fail_allCh_1D->setBinLabel(1, "GE-11", 1);
  me_GEM_fail_allCh_1D->setBinLabel(2, "GE11", 1);
  me_GEM_fail_allCh_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_chamber_1D->setAxisTitle("Chamber", 1);
  me_GEM_pass_chamber_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_chamber_1D->setAxisTitle("Chamber", 1);
  me_GEM_fail_chamber_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_chamber_p1_1D->setAxisTitle("Chamber", 1);
  me_GEM_pass_chamber_p1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_chamber_p1_1D->setAxisTitle("Chamber", 1);
  me_GEM_fail_chamber_p1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_chamber_p2_1D->setAxisTitle("Chamber", 1);
  me_GEM_pass_chamber_p2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_chamber_p2_1D->setAxisTitle("Chamber", 1);
  me_GEM_fail_chamber_p2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_chamber_n1_1D->setAxisTitle("Chamber", 1);
  me_GEM_pass_chamber_n1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_chamber_n1_1D->setAxisTitle("Chamber", 1);
  me_GEM_fail_chamber_n1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_chamber_n2_1D->setAxisTitle("Chamber", 1);
  me_GEM_pass_chamber_n2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_chamber_n2_1D->setAxisTitle("Chamber", 1);
  me_GEM_fail_chamber_n2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_pt_1D->setAxisTitle("P_{T}", 1);
  me_GEM_pass_pt_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_pt_1D->setAxisTitle("P_{T}", 1);
  me_GEM_fail_pt_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_eta_1D->setAxisTitle("#eta", 1);
  me_GEM_pass_eta_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_eta_1D->setAxisTitle("#eta", 1);
  me_GEM_fail_eta_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_phi_1D->setAxisTitle("#phi", 1);
  me_GEM_pass_phi_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_phi_1D->setAxisTitle("#phi", 1);
  me_GEM_fail_phi_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_pt_p1_1D->setAxisTitle("P_{T}", 1);
  me_GEM_pass_pt_p1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_pt_p1_1D->setAxisTitle("P_{T}", 1);
  me_GEM_fail_pt_p1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_eta_p1_1D->setAxisTitle("#eta", 1);
  me_GEM_pass_eta_p1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_eta_p1_1D->setAxisTitle("#eta", 1);
  me_GEM_fail_eta_p1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_phi_p1_1D->setAxisTitle("#phi", 1);
  me_GEM_pass_phi_p1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_phi_p1_1D->setAxisTitle("#phi", 1);
  me_GEM_fail_phi_p1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_pt_p2_1D->setAxisTitle("P_{T}", 1);
  me_GEM_pass_pt_p2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_pt_p2_1D->setAxisTitle("P_{T}", 1);
  me_GEM_fail_pt_p2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_eta_p2_1D->setAxisTitle("#eta", 1);
  me_GEM_pass_eta_p2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_eta_p2_1D->setAxisTitle("#eta", 1);
  me_GEM_fail_eta_p2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_phi_p2_1D->setAxisTitle("#phi", 1);
  me_GEM_pass_phi_p2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_phi_p2_1D->setAxisTitle("#phi", 1);
  me_GEM_fail_phi_p2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_pt_n1_1D->setAxisTitle("P_{T}", 1);
  me_GEM_pass_pt_n1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_pt_n1_1D->setAxisTitle("P_{T}", 1);
  me_GEM_fail_pt_n1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_eta_n1_1D->setAxisTitle("#eta", 1);
  me_GEM_pass_eta_n1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_eta_n1_1D->setAxisTitle("#eta", 1);
  me_GEM_fail_eta_n1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_phi_n1_1D->setAxisTitle("#phi", 1);
  me_GEM_pass_phi_n1_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_phi_n1_1D->setAxisTitle("#phi", 1);
  me_GEM_fail_phi_n1_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_pt_n2_1D->setAxisTitle("P_{T}", 1);
  me_GEM_pass_pt_n2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_pt_n2_1D->setAxisTitle("P_{T}", 1);
  me_GEM_fail_pt_n2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_eta_n2_1D->setAxisTitle("#eta", 1);
  me_GEM_pass_eta_n2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_eta_n2_1D->setAxisTitle("#eta", 1);
  me_GEM_fail_eta_n2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_pass_phi_n2_1D->setAxisTitle("#phi", 1);
  me_GEM_pass_phi_n2_1D->setAxisTitle("Number of passing probes", 2);
  me_GEM_fail_phi_n2_1D->setAxisTitle("#phi", 1);
  me_GEM_fail_phi_n2_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_fail_Ch_region->setBinLabel(1, "GE-11", 1);
  me_GEM_fail_Ch_region->setBinLabel(2, "GE11", 1);
  for (int i = 1; i < 37; ++i) {
    me_GEM_fail_Ch_region->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_fail_Ch_region->setAxisTitle("Chamber", 2);
  me_GEM_fail_Ch_region->setAxisTitle("Number of failing probes", 3);

  me_GEM_pass_Ch_region->setBinLabel(1, "GE-11", 1);
  me_GEM_pass_Ch_region->setBinLabel(2, "GE11", 1);
  for (int i = 1; i < 37; ++i) {
    me_GEM_pass_Ch_region->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_pass_Ch_region->setAxisTitle("Chamber", 2);
  me_GEM_pass_Ch_region->setAxisTitle("Number of passing probes", 3);

  me_GEM_fail_Ch_region_GE1->setBinLabel(1, "GE-1/1_L2", 1);
  me_GEM_fail_Ch_region_GE1->setBinLabel(2, "GE-1/1_L1", 1);
  me_GEM_fail_Ch_region_GE1->setBinLabel(3, "GE1/1_L1", 1);
  me_GEM_fail_Ch_region_GE1->setBinLabel(4, "GE1/1_L2", 1);
  for (int i = 1; i < 37; ++i) {
    me_GEM_fail_Ch_region_GE1->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_fail_Ch_region_GE1->setAxisTitle("Chamber", 2);
  me_GEM_fail_Ch_region_GE1->setAxisTitle("Number of passing probes", 3);

  me_GEM_pass_Ch_region_GE1->setBinLabel(1, "GE-1/1_L2", 1);
  me_GEM_pass_Ch_region_GE1->setBinLabel(2, "GE-1/1_L1", 1);
  me_GEM_pass_Ch_region_GE1->setBinLabel(3, "GE1/1_L1", 1);
  me_GEM_pass_Ch_region_GE1->setBinLabel(4, "GE1/1_L2", 1);
  for (int i = 1; i < 37; ++i) {
    me_GEM_pass_Ch_region_GE1->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_pass_Ch_region_GE1->setAxisTitle("Chamber", 2);
  me_GEM_pass_Ch_region_GE1->setAxisTitle("Number of passing probes", 3);

  me_GEM_fail_Ch_region_GE1_NoL->setBinLabel(1, "GE-1", 1);
  me_GEM_fail_Ch_region_GE1_NoL->setBinLabel(2, "GE+1", 1);
  for (int i = 1; i < 37; ++i) {
    me_GEM_fail_Ch_region_GE1_NoL->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_fail_Ch_region_GE1_NoL->setAxisTitle("Chamber", 2);
  me_GEM_fail_Ch_region_GE1_NoL->setAxisTitle("Number of passing probes", 3);

  me_GEM_pass_Ch_region_GE1_NoL->setBinLabel(1, "GE-1", 1);
  me_GEM_pass_Ch_region_GE1_NoL->setBinLabel(2, "GE+1", 1);
  for (int i = 1; i < 37; ++i) {
    me_GEM_pass_Ch_region_GE1_NoL->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_pass_Ch_region_GE1_NoL->setAxisTitle("Chamber", 2);
  me_GEM_pass_Ch_region_GE1_NoL->setAxisTitle("Number of passing probes", 3);

  for (int i = 1; i < 37; ++i) {
    me_GEM_fail_Ch_eta->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_fail_Ch_eta->setAxisTitle("#eta", 1);
  me_GEM_fail_Ch_eta->setAxisTitle("Chamber", 2);
  me_GEM_fail_Ch_eta->setAxisTitle("Number of failing probes", 3);

  for (int i = 1; i < 37; ++i) {
    me_GEM_pass_Ch_eta->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_pass_Ch_eta->setAxisTitle("#eta", 1);
  me_GEM_pass_Ch_eta->setAxisTitle("Chamber", 2);
  me_GEM_pass_Ch_eta->setAxisTitle("Number of passing probes", 3);

  for (int i = 1; i < 37; ++i) {
    me_GEM_fail_Ch_phi->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_fail_Ch_phi->setAxisTitle("#phi", 1);
  me_GEM_fail_Ch_phi->setAxisTitle("Chamber", 2);
  me_GEM_fail_Ch_phi->setAxisTitle("Number of failing probes", 3);

  for (int i = 1; i < 37; ++i) {
    me_GEM_pass_Ch_phi->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_pass_Ch_phi->setAxisTitle("#phi", 1);
  me_GEM_pass_Ch_phi->setAxisTitle("Chamber", 2);
  me_GEM_pass_Ch_phi->setAxisTitle("Number of passing probes", 3);

  for (int i = 1; i < 19; ++i) {
    me_ME0_pass_chamber_1D->setBinLabel(i, std::to_string(i), 1);
  }
  me_ME0_pass_chamber_1D->setAxisTitle("Chamber", 1);
  me_ME0_pass_chamber_1D->setAxisTitle("Number of passing probes", 2);
  for (int i = 1; i < 19; ++i) {
    me_ME0_fail_chamber_1D->setBinLabel(i, std::to_string(i), 1);
  }
  me_ME0_fail_chamber_1D->setAxisTitle("Chamber", 1);
  me_ME0_fail_chamber_1D->setAxisTitle("Number of failing probes", 2);

  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(1, "GE-2/1_L2", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(2, "GE-2/1_L1", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(3, "GE-1/1_L2", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(4, "GE-1/1_L1", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(5, "GE0-1", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(6, "GE0+1", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(7, "GE1/1_L1", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(8, "GE1/1_L2", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(9, "GE2/1_L1", 1);
  me_GEM_fail_Ch_region_layer_phase2->setBinLabel(10, "GE2/1_L2", 1);
  for (int i = 1; i < 37; ++i) {
    me_GEM_fail_Ch_region_layer_phase2->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_fail_Ch_region_layer_phase2->setAxisTitle("Chamber", 2);
  me_GEM_fail_Ch_region_layer_phase2->setAxisTitle("Number of passing probes", 3);

  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(1, "GE-2/1_L2", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(2, "GE-2/1_L1", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(3, "GE-1/1_L2", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(4, "GE-1/1_L1", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(5, "GE0-1", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(6, "GE0+1", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(7, "GE1/1_L1", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(8, "GE1/1_L2", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(9, "GE2/1_L1", 1);
  me_GEM_pass_Ch_region_layer_phase2->setBinLabel(10, "GE2/1_L2", 1);

  for (int i = 1; i < 37; ++i) {
    me_GEM_pass_Ch_region_layer_phase2->setBinLabel(i, std::to_string(i), 2);
  }
  me_GEM_pass_Ch_region_layer_phase2->setAxisTitle("Chamber", 2);
  me_GEM_pass_Ch_region_layer_phase2->setAxisTitle("Number of passing probes", 3);

  m_histos["GEM_nPassingProbe_Ch_region"] = me_GEM_pass_Ch_region;
  m_histos["GEM_nFailingProbe_Ch_region"] = me_GEM_fail_Ch_region;
  m_histos["GEM_nPassingProbe_Ch_region_GE1"] = me_GEM_pass_Ch_region_GE1;
  m_histos["GEM_nFailingProbe_Ch_region_GE1"] = me_GEM_fail_Ch_region_GE1;
  m_histos["GEM_nPassingProbe_Ch_region_GE1_NoL"] = me_GEM_pass_Ch_region_GE1_NoL;
  m_histos["GEM_nFailingProbe_Ch_region_GE1_NoL"] = me_GEM_fail_Ch_region_GE1_NoL;
  m_histos["GEM_nPassingProbe_Ch_eta"] = me_GEM_pass_Ch_eta;
  m_histos["GEM_nFailingProbe_Ch_eta"] = me_GEM_fail_Ch_eta;
  m_histos["GEM_nPassingProbe_Ch_phi"] = me_GEM_pass_Ch_phi;
  m_histos["GEM_nFailingProbe_Ch_phi"] = me_GEM_fail_Ch_phi;
  m_histos["GEM_nPassingProbe_allCh_1D"] = me_GEM_pass_allCh_1D;
  m_histos["GEM_nFailingProbe_allCh_1D"] = me_GEM_fail_allCh_1D;
  m_histos["GEM_nPassingProbe_chamber_1D"] = me_GEM_pass_chamber_1D;
  m_histos["GEM_nFailingProbe_chamber_1D"] = me_GEM_fail_chamber_1D;
  m_histos["GEM_nPassingProbe_chamber_p1_1D"] = me_GEM_pass_chamber_p1_1D;
  m_histos["GEM_nFailingProbe_chamber_p1_1D"] = me_GEM_fail_chamber_p1_1D;
  m_histos["GEM_nPassingProbe_chamber_p2_1D"] = me_GEM_pass_chamber_p2_1D;
  m_histos["GEM_nFailingProbe_chamber_p2_1D"] = me_GEM_fail_chamber_p2_1D;
  m_histos["GEM_nPassingProbe_chamber_n1_1D"] = me_GEM_pass_chamber_n1_1D;
  m_histos["GEM_nFailingProbe_chamber_n1_1D"] = me_GEM_fail_chamber_n1_1D;
  m_histos["GEM_nPassingProbe_chamber_n2_1D"] = me_GEM_pass_chamber_n2_1D;
  m_histos["GEM_nFailingProbe_chamber_n2_1D"] = me_GEM_fail_chamber_n2_1D;
  m_histos["GEM_nPassingProbe_pt_1D"] = me_GEM_pass_pt_1D;
  m_histos["GEM_nFailingProbe_pt_1D"] = me_GEM_fail_pt_1D;
  m_histos["GEM_nPassingProbe_eta_1D"] = me_GEM_pass_eta_1D;
  m_histos["GEM_nFailingProbe_eta_1D"] = me_GEM_fail_eta_1D;
  m_histos["GEM_nPassingProbe_phi_1D"] = me_GEM_pass_phi_1D;
  m_histos["GEM_nFailingProbe_phi_1D"] = me_GEM_fail_phi_1D;
  m_histos["GEM_nPassingProbe_pt_p1_1D"] = me_GEM_pass_pt_p1_1D;
  m_histos["GEM_nFailingProbe_pt_p1_1D"] = me_GEM_fail_pt_p1_1D;
  m_histos["GEM_nPassingProbe_eta_p1_1D"] = me_GEM_pass_eta_p1_1D;
  m_histos["GEM_nFailingProbe_eta_p1_1D"] = me_GEM_fail_eta_p1_1D;
  m_histos["GEM_nPassingProbe_phi_p1_1D"] = me_GEM_pass_phi_p1_1D;
  m_histos["GEM_nFailingProbe_phi_p1_1D"] = me_GEM_fail_phi_p1_1D;
  m_histos["GEM_nPassingProbe_pt_p2_1D"] = me_GEM_pass_pt_p2_1D;
  m_histos["GEM_nFailingProbe_pt_p2_1D"] = me_GEM_fail_pt_p2_1D;
  m_histos["GEM_nPassingProbe_eta_p2_1D"] = me_GEM_pass_eta_p2_1D;
  m_histos["GEM_nFailingProbe_eta_p2_1D"] = me_GEM_fail_eta_p2_1D;
  m_histos["GEM_nPassingProbe_phi_p2_1D"] = me_GEM_pass_phi_p2_1D;
  m_histos["GEM_nFailingProbe_phi_p2_1D"] = me_GEM_fail_phi_p2_1D;
  m_histos["GEM_nPassingProbe_pt_n1_1D"] = me_GEM_pass_pt_n1_1D;
  m_histos["GEM_nFailingProbe_pt_n1_1D"] = me_GEM_fail_pt_n1_1D;
  m_histos["GEM_nPassingProbe_eta_n1_1D"] = me_GEM_pass_eta_n1_1D;
  m_histos["GEM_nFailingProbe_eta_n1_1D"] = me_GEM_fail_eta_n1_1D;
  m_histos["GEM_nPassingProbe_phi_n1_1D"] = me_GEM_pass_phi_n1_1D;
  m_histos["GEM_nFailingProbe_phi_n1_1D"] = me_GEM_fail_phi_n1_1D;
  m_histos["GEM_nPassingProbe_pt_n2_1D"] = me_GEM_pass_pt_n2_1D;
  m_histos["GEM_nFailingProbe_pt_n2_1D"] = me_GEM_fail_pt_n2_1D;
  m_histos["GEM_nPassingProbe_eta_n2_1D"] = me_GEM_pass_eta_n2_1D;
  m_histos["GEM_nFailingProbe_eta_n2_1D"] = me_GEM_fail_eta_n2_1D;
  m_histos["GEM_nPassingProbe_phi_n2_1D"] = me_GEM_pass_phi_n2_1D;
  m_histos["GEM_nFailingProbe_phi_n2_1D"] = me_GEM_fail_phi_n2_1D;
  m_histos["ME0_nPassingProbe_chamber_1D"] = me_ME0_pass_chamber_1D;
  m_histos["ME0_nFailingProbe_chamber_1D"] = me_ME0_fail_chamber_1D;
  m_histos["GEM_nPassingProbe_Ch_region_layer_phase2"] = me_GEM_pass_Ch_region_layer_phase2;
  m_histos["GEM_nFailingProbe_Ch_region_layer_phase2"] = me_GEM_fail_Ch_region_layer_phase2;

  std::string baseDir_ = topFolder() + "/detailed/";
  iBooker.setCurrentFolder(baseDir_);
  m_histos["GEMhit_dx"] = iBooker.book1D("GEMhit_dx", "GEMhit_dx;probe dx [cm];Events", 100, 0., 10.);
  m_histos["GEMseg_dx"] = iBooker.book1D("GEMseg_dx", "GEMseg_dx;probe dx [cm];Events", 100, 0., 20.);

  m_histos["GEMhit_x"] = iBooker.book1D("GEMhit_x", "GEMhit_x;probe x [cm];Events", 100, -10., 10.);
  m_histos["GEMhit_x_GE2"] = iBooker.book1D("GEMhit_x_GE2", "GEMhit_x;probe x [cm];Events", 100, -10., 10.);
  m_histos["Cham_x"] = iBooker.book1D("Cham_x", "Cham_x;probe x [cm];Events", 100, -10., 10.);
  m_histos["Cham_x_GE2"] = iBooker.book1D("Cham_x_GE2", "Cham_x;probe x [cm];Events", 100, -10., 10.);

  m_histos["GEMhit_dx_GE2"] = iBooker.book1D("GEMhit_dx_GE2", "GEMhit_dx;probe dx [cm];Events", 100, 0., 10.);
}

void GEMTnPEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& context) {
  BaseTnPEfficiencyTask::analyze(event, context);

  edm::Handle<reco::MuonCollection> muons;
  event.getByToken(m_muToken, muons);

  //GEM variables
  std::vector<std::vector<int>> probe_coll_GEM_region;
  std::vector<std::vector<int>> probe_coll_GEM_lay;
  std::vector<std::vector<int>> probe_coll_GEM_chamber;
  std::vector<std::vector<float>> probe_coll_GEM_pt;
  std::vector<std::vector<float>> probe_coll_GEM_eta;
  std::vector<std::vector<float>> probe_coll_GEM_phi;
  std::vector<std::vector<int>> probe_coll_GEM_sta;
  std::vector<std::vector<float>> probe_coll_GEM_dx;
  std::vector<uint8_t> probe_coll_GEM_staMatch;

  //ME0 variables
  std::vector<std::vector<int>> probe_coll_ME0_region;
  std::vector<std::vector<int>> probe_coll_ME0_roll;
  std::vector<std::vector<int>> probe_coll_ME0_lay;
  std::vector<std::vector<int>> probe_coll_ME0_chamber;
  std::vector<std::vector<float>> probe_coll_ME0_pt;
  std::vector<std::vector<float>> probe_coll_ME0_eta;
  std::vector<std::vector<float>> probe_coll_ME0_phi;
  std::vector<std::vector<int>> probe_coll_ME0_sta;
  std::vector<std::vector<float>> probe_coll_ME0_dx;
  std::vector<uint8_t> probe_coll_ME0_staMatch;

  std::vector<unsigned> probe_indices;
  if (!m_probeIndices.empty())
    probe_indices = m_probeIndices.back();

  //Fill probe dx + subdetector coordinates
  for (const auto i : probe_indices) {
    //GEM variables
    std::vector<int> probe_GEM_region;
    std::vector<int> probe_GEM_sta;
    std::vector<int> probe_GEM_lay;
    std::vector<int> probe_GEM_chamber;
    std::vector<float> probe_GEM_pt;
    std::vector<float> probe_GEM_eta;
    std::vector<float> probe_GEM_phi;
    std::vector<float> probe_GEM_dx;
    //std::vector<float> probe_GEM_dx_seg;
    uint8_t GEM_stationMatching = 0;
    //ME0 variables
    std::vector<int> probe_ME0_region;
    std::vector<int> probe_ME0_roll;
    std::vector<int> probe_ME0_sta;
    std::vector<int> probe_ME0_lay;
    std::vector<int> probe_ME0_chamber;
    std::vector<float> probe_ME0_pt;
    std::vector<float> probe_ME0_eta;
    std::vector<float> probe_ME0_phi;
    std::vector<float> probe_ME0_dx;
    uint8_t ME0_stationMatching = 0;

    bool gem_matched = false;  // fill detailed plots only for probes matching GEM

    for (const auto& chambMatch : (*muons).at(i).matches()) {
      // look in GEMs
      if (chambMatch.detector() == MuonSubdetId::GEM) {
        if (chambMatch.edgeX < m_borderCut && chambMatch.edgeY < m_borderCut) {
          gem_matched = true;  //fill detailed plots if at least one GEM match

          GEMDetId chId(chambMatch.id.rawId());

          const int roll = chId.roll();
          const int region = chId.region();
          const int station = chId.station();
          const int layer = chId.layer();
          const int chamber = chId.chamber();
          const float pt = (*muons).at(i).pt();
          const float eta = (*muons).at(i).eta();
          const float phi = (*muons).at(i).phi();

          reco::MuonGEMHitMatch closest_matchedHit;
          double smallestDx = 99999.;
          double matched_GEMHit_x = 99999.;

          for (auto& gemHit : chambMatch.gemHitMatches) {
            float dx = std::abs(chambMatch.x - gemHit.x);
            if (dx < smallestDx) {
              smallestDx = dx;
              closest_matchedHit = gemHit;
              matched_GEMHit_x = gemHit.x;
            }
          }

          reco::MuonSegmentMatch closest_matchedSegment;
          double smallestDx_seg = 99999.;

          for (auto& seg : chambMatch.gemMatches) {
            float dx_seg = std::abs(chambMatch.x - seg.x);
            if (dx_seg < smallestDx_seg) {
              smallestDx_seg = dx_seg;
              closest_matchedSegment = seg;
            }
          }
          if (m_detailedAnalysis && gem_matched) {
            m_histos.find("GEMhit_dx")->second->Fill(smallestDx);
            m_histos.find("GEMhit_x")->second->Fill(matched_GEMHit_x);
            m_histos.find("Cham_x")->second->Fill(chambMatch.x);
            m_histos.find("GEMseg_dx")->second->Fill(smallestDx_seg);
            if (station == 2) {
              m_histos.find("GEMhit_dx_GE2")->second->Fill(smallestDx);
              m_histos.find("GEMhit_x_GE2")->second->Fill(matched_GEMHit_x);
              m_histos.find("Cham_x_GE2")->second->Fill(chambMatch.x);
            }
          }

          GEM_stationMatching = GEM_stationMatching | (1 << (station - 1));

          probe_GEM_region.push_back(region);
          probe_GEM_sta.push_back(station);
          probe_GEM_lay.push_back(layer);
          probe_GEM_chamber.push_back(chamber);
          probe_GEM_pt.push_back(pt);
          probe_GEM_eta.push_back(eta);
          probe_GEM_phi.push_back(phi);
          probe_GEM_dx.push_back(smallestDx);
          //probe_GEM_dx.push_back(smallestDx_seg);
          //probe_GEM_dx_seg.push_back(smallestDx_seg);

          if (station == 0) {
            reco::MuonSegmentMatch closest_matchedSegment_ME0;
            double smallestDx_ME0 = 99999.;
            for (auto& seg : chambMatch.gemMatches) {
              float dx = std::abs(chambMatch.x - seg.x);
              if (dx < smallestDx_ME0) {
                smallestDx_ME0 = dx;
                closest_matchedSegment_ME0 = seg;
              }
            }
            ME0_stationMatching = ME0_stationMatching | (1 << (station - 1));
            probe_ME0_region.push_back(region);
            probe_ME0_roll.push_back(roll);
            probe_ME0_sta.push_back(station);
            probe_ME0_lay.push_back(layer);
            probe_ME0_chamber.push_back(chamber);
            probe_ME0_pt.push_back(pt);
            probe_ME0_eta.push_back(eta);
            probe_ME0_phi.push_back(phi);
            probe_ME0_dx.push_back(smallestDx_ME0);
          }
        }
      } else
        continue;
    }  //loop over chamber matches

    //Fill detailed plots
    if (m_detailedAnalysis && gem_matched) {
      m_histos.find("probeEta")->second->Fill(abs((*muons).at(i).eta()));
      m_histos.find("probePhi")->second->Fill((*muons).at(i).phi());
      m_histos.find("probeNumberOfMatchedStations")->second->Fill((*muons).at(i).numberOfMatchedStations());
      m_histos.find("probePt")->second->Fill((*muons).at(i).pt());
      //for(int ii=0; i<probe_GEM_dx.size(); ii++)
      //{
      //    m_histos.find("GEMhit_dx")->second->Fill(probe_GEM_dx[ii]);
      //    m_histos.find("GEMseg_dx")->second->Fill(probe_GEM_dx_seg[ii]);
      //}
    }

    //Fill GEM variables
    probe_coll_GEM_region.push_back(probe_GEM_region);
    probe_coll_GEM_sta.push_back(probe_GEM_sta);
    probe_coll_GEM_lay.push_back(probe_GEM_lay);
    probe_coll_GEM_chamber.push_back(probe_GEM_chamber);
    probe_coll_GEM_pt.push_back(probe_GEM_pt);
    probe_coll_GEM_eta.push_back(probe_GEM_eta);
    probe_coll_GEM_phi.push_back(probe_GEM_phi);
    probe_coll_GEM_dx.push_back(probe_GEM_dx);
    probe_coll_GEM_staMatch.push_back(GEM_stationMatching);

    //Fill ME0 variables
    probe_coll_ME0_region.push_back(probe_ME0_region);
    probe_coll_ME0_roll.push_back(probe_ME0_roll);
    probe_coll_ME0_sta.push_back(probe_ME0_sta);
    probe_coll_ME0_lay.push_back(probe_ME0_lay);
    probe_coll_ME0_chamber.push_back(probe_ME0_chamber);
    probe_coll_ME0_pt.push_back(probe_ME0_pt);
    probe_coll_ME0_eta.push_back(probe_ME0_eta);
    probe_coll_ME0_phi.push_back(probe_ME0_phi);
    probe_coll_ME0_dx.push_back(probe_ME0_dx);
    probe_coll_ME0_staMatch.push_back(ME0_stationMatching);

  }  //loop over probe collection

  //Loop over probes
  for (unsigned i = 0; i < probe_indices.size(); ++i) {
    uint8_t GEM_matchPatt = probe_coll_GEM_staMatch.at(i);
    //uint8_t ME0_matchPatt = probe_coll_ME0_staMatch.at(i);

    //Loop over ME0 matches
    unsigned nME0_matches = probe_coll_ME0_region.at(i).size();
    for (unsigned j = 0; j < nME0_matches; ++j) {
      //ME0 variables
      int ME0_region = probe_coll_ME0_region.at(i).at(j);
      //int ME0_roll   = probe_coll_ME0_roll.at(i).at(j);
      int ME0_sta = probe_coll_ME0_sta.at(i).at(j);
      //int ME0_lay    = probe_coll_ME0_lay.at(i).at(j);
      int ME0_chamber = probe_coll_ME0_chamber.at(i).at(j);
      //float ME0_pt   = probe_coll_ME0_pt.at(i).at(j);
      float ME0_dx = probe_coll_ME0_dx.at(i).at(j);
      //float ME0_eta   = probe_coll_ME0_eta.at(i).at(j);
      //float ME0_phi   = probe_coll_ME0_phi.at(i).at(j);

      if (ME0_sta == 0) {
        if (ME0_dx < m_dxCut) {
          m_histos.find("ME0_nPassingProbe_chamber_1D")->second->Fill(ME0_chamber);
          if (ME0_region < 0)
            m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(4, ME0_chamber);
          else if (ME0_region > 0)
            m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(5, ME0_chamber);
        } else {
          m_histos.find("ME0_nFailingProbe_chamber_1D")->second->Fill(ME0_chamber);
          if (ME0_region < 0)
            m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(4, ME0_chamber);
          else if (ME0_region > 0)
            m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(5, ME0_chamber);
        }
      }
    }
    //

    //Loop over GEM matches
    unsigned nGEM_matches = probe_coll_GEM_region.at(i).size();
    for (unsigned j = 0; j < nGEM_matches; ++j) {
      //GEM variables
      int GEM_region = probe_coll_GEM_region.at(i).at(j);
      int GEM_sta = probe_coll_GEM_sta.at(i).at(j);
      int GEM_lay = probe_coll_GEM_lay.at(i).at(j);
      int GEM_chamber = probe_coll_GEM_chamber.at(i).at(j);
      float GEM_pt = probe_coll_GEM_pt.at(i).at(j);
      float GEM_dx = probe_coll_GEM_dx.at(i).at(j);
      float GEM_eta = probe_coll_GEM_eta.at(i).at(j);
      float GEM_phi = probe_coll_GEM_phi.at(i).at(j);

      //Fill GEM plots
      if (((GEM_matchPatt & (1 << (GEM_sta - 1))) != 0) && GEM_sta != 0)  //avoids 0 station matching
      {
        if (GEM_dx < m_dxCut) {
          if (GEM_region == 1 && GEM_lay == 0 && GEM_sta == 2 && GEM_chamber == 16)
            continue;  //exclude GE2 ch16 of Run3
          m_histos.find("GEM_nPassingProbe_Ch_region")->second->Fill(GEM_region, GEM_chamber);
          m_histos.find("GEM_nPassingProbe_Ch_eta")->second->Fill(abs(GEM_eta), GEM_chamber);
          m_histos.find("GEM_nPassingProbe_Ch_phi")->second->Fill(GEM_phi, GEM_chamber);
          m_histos.find("GEM_nPassingProbe_allCh_1D")->second->Fill(GEM_region);
          m_histos.find("GEM_nPassingProbe_chamber_1D")->second->Fill(GEM_chamber);
          if (GEM_region < 0) {
            if (GEM_sta == 2 and GEM_lay == 2)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(0, GEM_chamber);
            else if (GEM_sta == 2 and GEM_lay == 1)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(1, GEM_chamber);
            else if (GEM_sta == 1 and GEM_lay == 2)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(2, GEM_chamber);
            else if (GEM_sta == 1 and GEM_lay == 1)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(3, GEM_chamber);
          }
          if (GEM_region > 0) {
            if (GEM_sta == 1 and GEM_lay == 1)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(6, GEM_chamber);
            else if (GEM_sta == 1 and GEM_lay == 2)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(7, GEM_chamber);
            else if (GEM_sta == 2 and GEM_lay == 1)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(8, GEM_chamber);
            else if (GEM_sta == 2 and GEM_lay == 2)
              m_histos.find("GEM_nPassingProbe_Ch_region_layer_phase2")->second->Fill(9, GEM_chamber);
          }
          if (GEM_region == -1 && GEM_sta == 1) {
            m_histos.find("GEM_nPassingProbe_Ch_region_GE1_NoL")->second->Fill(0, GEM_chamber);
          } else if (GEM_region == 1 && GEM_sta == 1) {
            m_histos.find("GEM_nPassingProbe_Ch_region_GE1_NoL")->second->Fill(1, GEM_chamber);
          }

          if (GEM_region == 1 && GEM_lay == 1 && GEM_sta == 1) {
            m_histos.find("GEM_nPassingProbe_chamber_p1_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nPassingProbe_Ch_region_GE1")->second->Fill(2, GEM_chamber);
            m_histos.find("GEM_nPassingProbe_pt_p1_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nPassingProbe_eta_p1_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nPassingProbe_phi_p1_1D")->second->Fill(GEM_phi);
          } else if (GEM_region == 1 && GEM_lay == 2 && GEM_sta == 1) {
            m_histos.find("GEM_nPassingProbe_chamber_p2_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nPassingProbe_Ch_region_GE1")->second->Fill(3, GEM_chamber);
            m_histos.find("GEM_nPassingProbe_pt_p2_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nPassingProbe_eta_p2_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nPassingProbe_phi_p2_1D")->second->Fill(GEM_phi);
          } else if (GEM_region == -1 && GEM_lay == 1 && GEM_sta == 1) {
            m_histos.find("GEM_nPassingProbe_chamber_n1_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nPassingProbe_Ch_region_GE1")->second->Fill(1, GEM_chamber);
            m_histos.find("GEM_nPassingProbe_pt_n1_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nPassingProbe_eta_n1_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nPassingProbe_phi_n1_1D")->second->Fill(GEM_phi);
          } else if (GEM_region == -1 && GEM_lay == 2 && GEM_sta == 1) {
            m_histos.find("GEM_nPassingProbe_chamber_n2_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nPassingProbe_Ch_region_GE1")->second->Fill(0, GEM_chamber);
            m_histos.find("GEM_nPassingProbe_pt_n2_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nPassingProbe_eta_n2_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nPassingProbe_phi_n2_1D")->second->Fill(GEM_phi);
          }
          m_histos.find("GEM_nPassingProbe_pt_1D")->second->Fill(GEM_pt);
          m_histos.find("GEM_nPassingProbe_eta_1D")->second->Fill(abs(GEM_eta));
          m_histos.find("GEM_nPassingProbe_phi_1D")->second->Fill(GEM_phi);
        } else {
          if (GEM_region == 1 && GEM_lay == 0 && GEM_sta == 2 && GEM_chamber == 16)
            continue;
          m_histos.find("GEM_nFailingProbe_Ch_region")->second->Fill(GEM_region, GEM_chamber);
          m_histos.find("GEM_nFailingProbe_Ch_eta")->second->Fill(abs(GEM_eta), GEM_chamber);
          m_histos.find("GEM_nFailingProbe_Ch_phi")->second->Fill(GEM_phi, GEM_chamber);
          m_histos.find("GEM_nFailingProbe_allCh_1D")->second->Fill(GEM_region);
          m_histos.find("GEM_nFailingProbe_chamber_1D")->second->Fill(GEM_chamber);
          if (GEM_region < 0) {
            if (GEM_sta == 2 and GEM_lay == 2)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(0, GEM_chamber);
            else if (GEM_sta == 2 and GEM_lay == 1)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(1, GEM_chamber);
            else if (GEM_sta == 1 and GEM_lay == 2)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(2, GEM_chamber);
            else if (GEM_sta == 1 and GEM_lay == 1)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(3, GEM_chamber);
          }
          if (GEM_region > 0) {
            if (GEM_sta == 1 and GEM_lay == 1)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(6, GEM_chamber);
            else if (GEM_sta == 1 and GEM_lay == 2)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(7, GEM_chamber);
            else if (GEM_sta == 2 and GEM_lay == 1)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(8, GEM_chamber);
            else if (GEM_sta == 2 and GEM_lay == 2)
              m_histos.find("GEM_nFailingProbe_Ch_region_layer_phase2")->second->Fill(9, GEM_chamber);
          }
          if (GEM_region == -1 && GEM_sta == 1) {
            m_histos.find("GEM_nFailingProbe_Ch_region_GE1_NoL")->second->Fill(0, GEM_chamber);
          } else if (GEM_region == 1 && GEM_sta == 1) {
            m_histos.find("GEM_nFailingProbe_Ch_region_GE1_NoL")->second->Fill(1, GEM_chamber);
          }
          //
          if (GEM_region == 1 && GEM_lay == 1 && GEM_sta == 1) {
            m_histos.find("GEM_nFailingProbe_chamber_p1_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nFailingProbe_Ch_region_GE1")->second->Fill(2, GEM_chamber);
            m_histos.find("GEM_nFailingProbe_pt_p1_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nFailingProbe_eta_p1_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nFailingProbe_phi_p1_1D")->second->Fill(GEM_phi);
          } else if (GEM_region == 1 && GEM_lay == 2 && GEM_sta == 1) {
            m_histos.find("GEM_nFailingProbe_chamber_p2_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nFailingProbe_Ch_region_GE1")->second->Fill(3, GEM_chamber);
            m_histos.find("GEM_nFailingProbe_pt_p2_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nFailingProbe_eta_p2_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nFailingProbe_phi_p2_1D")->second->Fill(GEM_phi);
          } else if (GEM_region == -1 && GEM_lay == 1 && GEM_sta == 1) {
            m_histos.find("GEM_nFailingProbe_chamber_n1_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nFailingProbe_Ch_region_GE1")->second->Fill(1, GEM_chamber);
            m_histos.find("GEM_nFailingProbe_pt_n1_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nFailingProbe_eta_n1_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nFailingProbe_phi_n1_1D")->second->Fill(GEM_phi);
          } else if (GEM_region == -1 && GEM_lay == 2 && GEM_sta == 1) {
            m_histos.find("GEM_nFailingProbe_chamber_n2_1D")->second->Fill(GEM_chamber);
            m_histos.find("GEM_nFailingProbe_Ch_region_GE1")->second->Fill(0, GEM_chamber);
            m_histos.find("GEM_nFailingProbe_pt_n2_1D")->second->Fill(GEM_pt);
            m_histos.find("GEM_nFailingProbe_eta_n2_1D")->second->Fill(abs(GEM_eta));
            m_histos.find("GEM_nFailingProbe_phi_n2_1D")->second->Fill(GEM_phi);
          }
          m_histos.find("GEM_nFailingProbe_pt_1D")->second->Fill(GEM_pt);
          m_histos.find("GEM_nFailingProbe_eta_1D")->second->Fill(abs(GEM_eta));
          m_histos.find("GEM_nFailingProbe_phi_1D")->second->Fill(GEM_phi);
        }
      }
    }
  }
}

std::string GEMTnPEfficiencyTask::topFolder() const { return "GEM/Segment_TnP/"; };

DEFINE_FWK_MODULE(GEMTnPEfficiencyTask);
