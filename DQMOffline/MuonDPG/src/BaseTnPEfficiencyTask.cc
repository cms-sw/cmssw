/*
 * \file BaseTnPEfficiencyTask.cc
 *
 * \author L. Lunerti - INFN Bologna
 *
 */

#include "DQMOffline/MuonDPG/interface/BaseTnPEfficiencyTask.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"

// Geometry
#include "DataFormats/GeometryVector/interface/Pi.h"

//Math
#include "DataFormats/Math/interface/deltaR.h"

//Root
#include "TRegexp.h"

#include <tuple>
#include <algorithm>

BaseTnPEfficiencyTask::BaseTnPEfficiencyTask(const edm::ParameterSet& config)
    : m_nEvents(0),
      m_muToken(consumes<reco::MuonCollection>(config.getUntrackedParameter<edm::InputTag>("inputTagMuons"))),
      m_borderCut(config.getUntrackedParameter<double>("borderCut")),
      m_dxCut(config.getUntrackedParameter<double>("dx_cut")),
      m_detailedAnalysis(config.getUntrackedParameter<bool>("detailedAnalysis")),
      m_primaryVerticesToken(
          consumes<std::vector<reco::Vertex>>(config.getUntrackedParameter<edm::InputTag>("inputTagPrimaryVertices"))),
      m_triggerResultsToken(
          consumes<edm::TriggerResults>(config.getUntrackedParameter<edm::InputTag>("trigResultsTag"))),
      m_triggerEventToken(consumes<trigger::TriggerEvent>(config.getUntrackedParameter<edm::InputTag>("trigEventTag"))),
      m_trigName(config.getUntrackedParameter<std::string>("trigName")),
      m_probeSelector(config.getUntrackedParameter<std::string>("probeCut")),
      m_dxyCut(config.getUntrackedParameter<double>("probeDxyCut")),
      m_dzCut(config.getUntrackedParameter<double>("probeDzCut")),
      m_tagSelector(config.getUntrackedParameter<std::string>("tagCut")),
      m_lowPairMassCut(config.getUntrackedParameter<double>("lowPairMassCut")),
      m_highPairMassCut(config.getUntrackedParameter<double>("highPairMassCut")) {
  LogTrace("DQMOffline|MuonDPG|BaseTnPEfficiencyTask") << "[BaseTnPEfficiencyTask]: Constructor" << std::endl;
}

BaseTnPEfficiencyTask::~BaseTnPEfficiencyTask() {
  LogTrace("DQMOffline|MuonDPG|BaseTnPEfficiencyTask")
      << "[BaseTnPEfficiencyTask]: analyzed " << m_nEvents << " events" << std::endl;
}

void BaseTnPEfficiencyTask::dqmBeginRun(const edm::Run& run, const edm::EventSetup& context) {
  bool changed = true;
  m_hltConfig.init(run, context, "HLT", changed);

  bool enableWildCard = true;

  TString tName = TString(m_trigName);
  TRegexp tNamePattern = TRegexp(tName, enableWildCard);

  for (unsigned iPath = 0; iPath < m_hltConfig.size(); ++iPath) {
    TString pathName = TString(m_hltConfig.triggerName(iPath));
    if (pathName.Contains(tNamePattern)) {
      m_trigIndices.push_back(static_cast<int>(iPath));
    }
  }
}

void BaseTnPEfficiencyTask::bookHistograms(DQMStore::IBooker& iBooker,
                                           edm::Run const& run,
                                           edm::EventSetup const& context) {
  LogTrace("DQMOffline|MuonDPG|BaseTnPEfficiencyTask") << "[BaseTnPEfficiencyTask]: bookHistograms" << std::endl;

  if (m_detailedAnalysis) {
    std::string baseDir = topFolder() + "/detailed/";
    iBooker.setCurrentFolder(baseDir);

    LogTrace("DQMOffline|MuonDPG|BaseTnPEfficiencyTask")
        << "[BaseTnPEfficiencyTask]: booking histos in " << baseDir << std::endl;

    m_histos["probePt"] = iBooker.book1D("probePt", "probePt;probe p_{T} [GeV];Events", 125, 0., 250.);
    m_histos["probeEta"] = iBooker.book1D("probeEta", "probeEta;probe #eta;Events", 24, -2.4, 2.4);
    m_histos["probePhi"] = iBooker.book1D("probePhi", "probePhi;probe #phi; Events", 36, -TMath::Pi(), TMath::Pi());
    m_histos["probeNumberOfMatchedStations"] = iBooker.book1D(
        "probeNumberOfMatchedStations", "probeNumberOfMatchedStations;Number of matched stations;Events", 5, 0., 5);
    m_histos["pairMass"] = iBooker.book1D("pairMass", "pairMass", 25, 70., 120.);
  }
}

void BaseTnPEfficiencyTask::analyze(const edm::Event& event, const edm::EventSetup& context) {
  ++m_nEvents;

  edm::Handle<reco::MuonCollection> muons;
  if (!event.getByToken(m_muToken, muons))
    return;

  edm::Handle<std::vector<reco::Vertex>> vtxs;
  if (!event.getByToken(m_primaryVerticesToken, vtxs))
    return;
  const reco::Vertex& vertex = vtxs->at(0);

  edm::Handle<edm::TriggerResults> triggerResults;
  if (!event.getByToken(m_triggerResultsToken, triggerResults))
    return;

  edm::Handle<trigger::TriggerEvent> triggerEvent;
  if (!event.getByToken(m_triggerEventToken, triggerEvent))
    return;

  //common tnp variables
  std::vector<unsigned> preSel_tag_indices;
  std::vector<unsigned> tag_indices;
  std::vector<unsigned> preSel_probe_indices;
  std::vector<unsigned> probe_indices;

  if (muons.isValid() && vtxs.isValid()) {
    //Is there a better way to initialize two different type variables?
    for (auto [muon, muonColl_index] = std::tuple{std::vector<reco::Muon>::const_iterator{(*muons).begin()}, 0};
         muon != (*muons).end();
         ++muon, ++muonColl_index) {
      bool trigMatch = false;

      //Getting trigger infos for tag selection
      if (triggerResults.isValid() && triggerEvent.isValid()) {
        const trigger::TriggerObjectCollection trigObjColl = triggerEvent->getObjects();
        trigMatch = hasTrigger(m_trigIndices, trigObjColl, triggerEvent, *muon);
      }

      //Probe selection
      if (m_probeSelector(*muon) && (std::abs(muon->muonBestTrack()->dxy(vertex.position())) < m_dxyCut) &&
          (std::abs(muon->muonBestTrack()->dz(vertex.position())) < m_dzCut)) {
        preSel_probe_indices.push_back(muonColl_index);
      }
      //Tag selection
      if (m_tagSelector(*muon) && trigMatch) {
        preSel_tag_indices.push_back(muonColl_index);
      }

    }  //loop over muons
  }

  //Probe selection
  for (const auto i_tag : preSel_tag_indices) {
    reco::Muon tag = (*muons).at(i_tag);
    float pt_max = 0.;
    int max_pt_idx;
    bool pair_found = false;

    for (const auto i_probe : preSel_probe_indices) {
      //Prevent tag and probe to be the same object
      if (i_probe == i_tag)
        continue;

      reco::Muon preSel_probe = (*muons).at(i_probe);

      int pair_charge_product = tag.charge() * preSel_probe.charge();

      //check if tag+probe pair is compatible with Z decay
      if (pair_charge_product > 0)
        continue;

      float pair_mass = (tag.polarP4() + preSel_probe.polarP4()).M();
      m_histos.find("pairMass")->second->Fill(pair_mass);

      if (pair_mass < m_lowPairMassCut || pair_mass > m_highPairMassCut)
        continue;

      float pair_pt = (tag.polarP4() + preSel_probe.polarP4()).Pt();
      if (pair_pt > pt_max) {
        pair_found = true;
        pt_max = pair_pt;
        max_pt_idx = i_probe;
      }
    }
    if (pair_found) {
      probe_indices.push_back(max_pt_idx);
      tag_indices.push_back(i_tag);
    }
  }

  m_probeIndices.push_back(probe_indices);
  m_tagIndices.push_back(tag_indices);
}

bool BaseTnPEfficiencyTask::hasTrigger(std::vector<int>& trigIndices,
                                       const trigger::TriggerObjectCollection& trigObjs,
                                       edm::Handle<trigger::TriggerEvent>& trigEvent,
                                       const reco::Muon& muon) {
  float dR2match = 999.;
  for (int trigIdx : trigIndices) {
    const std::vector<std::string> trigModuleLabels = m_hltConfig.moduleLabels(trigIdx);

    const unsigned trigModuleIndex =
        std::find(trigModuleLabels.begin(), trigModuleLabels.end(), "hltBoolEnd") - trigModuleLabels.begin() - 1;
    const unsigned hltFilterIndex = trigEvent->filterIndex(edm::InputTag(trigModuleLabels[trigModuleIndex], "", "HLT"));
    if (hltFilterIndex < trigEvent->sizeFilters()) {
      const trigger::Keys keys = trigEvent->filterKeys(hltFilterIndex);
      const trigger::Vids vids = trigEvent->filterIds(hltFilterIndex);
      const unsigned nTriggers = vids.size();

      for (unsigned iTrig = 0; iTrig < nTriggers; ++iTrig) {
        trigger::TriggerObject trigObj = trigObjs[keys[iTrig]];
        float dR2 = deltaR2(muon, trigObj);
        if (dR2 < dR2match)
          dR2match = dR2;
      }
    }
  }

  return dR2match < 0.01;
}
