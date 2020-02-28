// -*- C++ -*-
//
// Package:     HigPhotonJetHLTOfflineSource
// Class:       HigPhotonJetHLTOfflineSource
//

//
// Author: Xin Shi <Xin.Shi@cern.ch>
// Created: 2014.07.22
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <TLorentzVector.h>
#include <TH2F.h>

//  Define the interface
class HigPhotonJetHLTOfflineSource : public DQMOneEDAnalyzer<> {
public:
  explicit HigPhotonJetHLTOfflineSource(const edm::ParameterSet&);

private:
  // Analyzer Methods
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void dqmEndRun(const edm::Run&, const edm::EventSetup&) override;
  bool isMonitoredTriggerAccepted(const edm::TriggerNames&, const edm::Handle<edm::TriggerResults>&);

  // Input from Configuration File
  edm::ParameterSet pset_;
  std::string hltProcessName_;
  std::vector<std::string> hltPathsToCheck_;
  std::string dirname_;
  bool verbose_;
  bool triggerAccept_;

  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  edm::EDGetTokenT<reco::PhotonCollection> photonsToken_;
  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken_;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken_;

  double pfjetMinPt_;
  double photonMinPt_;

  // Member Variables

  MonitorElement* nvertices_reco_;
  MonitorElement* nvertices_;
  MonitorElement* nphotons_reco_;
  MonitorElement* nphotons_;
  MonitorElement* photonpt_reco_;
  MonitorElement* photonpt_;
  MonitorElement* photonrapidity_reco_;
  MonitorElement* photonrapidity_;
  MonitorElement* pfmet_reco_;
  MonitorElement* pfmet_;
  MonitorElement* pfmetphi_reco_;
  MonitorElement* pfmetphi_;
  MonitorElement* npfjets_reco_;
  MonitorElement* npfjets_;
  MonitorElement* delphiphomet_reco_;
  MonitorElement* delphiphomet_;
  MonitorElement* delphijetmet_reco_;
  MonitorElement* delphijetmet_;
  MonitorElement* invmassjj_reco_;
  MonitorElement* invmassjj_;
  MonitorElement* deletajj_reco_;
  MonitorElement* deletajj_;
  MonitorElement* triggers_reco_;
  MonitorElement* triggers_;
  MonitorElement* trigvsnvtx_reco_;
  MonitorElement* trigvsnvtx_;

  double evtsrun_;
};

// Class Methods

HigPhotonJetHLTOfflineSource::HigPhotonJetHLTOfflineSource(const edm::ParameterSet& pset) : pset_(pset) {
  hltProcessName_ = pset.getParameter<std::string>("hltProcessName");
  hltPathsToCheck_ = pset.getParameter<std::vector<std::string>>("hltPathsToCheck");
  verbose_ = pset.getUntrackedParameter<bool>("verbose", false);
  triggerAccept_ = pset.getUntrackedParameter<bool>("triggerAccept", true);
  triggerResultsToken_ = consumes<edm::TriggerResults>(pset.getParameter<edm::InputTag>("triggerResultsToken"));
  dirname_ = pset.getUntrackedParameter<std::string>("dirname", std::string("HLT/Higgs/PhotonJet/"));
  pvToken_ = consumes<reco::VertexCollection>(pset.getParameter<edm::InputTag>("pvToken"));
  photonsToken_ = consumes<reco::PhotonCollection>(pset.getParameter<edm::InputTag>("photonsToken"));
  pfMetToken_ = consumes<reco::PFMETCollection>(pset.getParameter<edm::InputTag>("pfMetToken"));
  pfJetsToken_ = consumes<reco::PFJetCollection>(pset.getParameter<edm::InputTag>("pfJetsToken"));
  pfjetMinPt_ = pset.getUntrackedParameter<double>("pfjetMinPt", 0.0);
  photonMinPt_ = pset.getUntrackedParameter<double>("photonMinPt", 0.0);
}

void HigPhotonJetHLTOfflineSource::dqmBeginRun(const edm::Run& iRun,
                                               const edm::EventSetup& iSetup) {  // Initialize hltConfig
  HLTConfigProvider hltConfig;
  bool changedConfig;
  if (!hltConfig.init(iRun, iSetup, hltProcessName_, changedConfig)) {
    edm::LogError("HLTPhotonJetVal") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  evtsrun_ = 0;
}

void HigPhotonJetHLTOfflineSource::bookHistograms(DQMStore::IBooker& iBooker,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  iBooker.setCurrentFolder(dirname_);
  nvertices_reco_ = iBooker.book1D("nvertices_reco", "Reco: Number of vertices", 100, 0, 100);
  nvertices_ = iBooker.book1D("nvertices", "Number of vertices", 100, 0, 100);
  nphotons_reco_ = iBooker.book1D("nphotons_reco", "Reco: Number of photons", 100, 0, 10);
  nphotons_ = iBooker.book1D("nphotons", "Number of photons", 100, 0, 10);
  photonpt_reco_ = iBooker.book1D("photonpt_reco", "Reco: Photons pT", 100, 0, 500);
  photonpt_ = iBooker.book1D("photonpt", "Photons pT", 100, 0, 500);
  photonrapidity_reco_ = iBooker.book1D("photonrapidity_reco", "Reco: Photons rapidity;y_{#gamma}", 100, -2.5, 2.5);
  photonrapidity_ = iBooker.book1D("photonrapidity", "Photons rapidity;y_{#gamma}", 100, -2.5, 2.5);
  pfmet_reco_ = iBooker.book1D("pfmet_reco", "Reco: PF MET", 100, 0, 250);
  pfmet_ = iBooker.book1D("pfmet", "PF MET", 100, 0, 250);
  pfmetphi_reco_ = iBooker.book1D("pfmetphi_reco", "Reco: PF MET phi;#phi_{PFMET}", 100, -4, 4);
  pfmetphi_ = iBooker.book1D("pfmetphi", "PF MET phi;#phi_{PFMET}", 100, -4, 4);
  delphiphomet_reco_ =
      iBooker.book1D("delphiphomet_reco", "Reco: #Delta#phi(photon, MET);#Delta#phi(#gamma,MET)", 100, 0, 4);
  delphiphomet_ = iBooker.book1D("delphiphomet", "#Delta#phi(photon, MET);#Delta#phi(#gamma,MET)", 100, 0, 4);
  npfjets_reco_ = iBooker.book1D("npfjets_reco", "Reco: Number of PF Jets", 100, 0, 20);
  npfjets_ = iBooker.book1D("npfjets", "Number of PF Jets", 100, 0, 20);
  delphijetmet_reco_ =
      iBooker.book1D("delphijetmet_reco", "Reco: #Delta#phi(PFJet, MET);#Delta#phi(Jet,MET)", 100, 0, 4);
  delphijetmet_ = iBooker.book1D("delphijetmet", "#Delta#phi(PFJet, MET);#Delta#phi(Jet,MET)", 100, 0, 4);
  invmassjj_reco_ = iBooker.book1D("invmassjj_reco", "Reco: Inv mass two leading jets;M_{jj}[GeV]", 100, 0, 2000);
  invmassjj_ = iBooker.book1D("invmassjj", "Inv mass two leading jets;M_{jj}[GeV]", 100, 0, 2000);
  deletajj_reco_ = iBooker.book1D("deletajj_reco", "Reco: #Delta#eta(jj);|#Delta#eta_{jj}|", 100, 0, 6);
  deletajj_ = iBooker.book1D("deletajj", "#Delta#eta(jj);|#Delta#eta_{jj}|", 100, 0, 6);
  triggers_reco_ =
      iBooker.book1D("triggers_reco", "Reco: Triggers", hltPathsToCheck_.size(), 0, hltPathsToCheck_.size());
  triggers_ = iBooker.book1D("triggers", "Triggers", hltPathsToCheck_.size(), 0, hltPathsToCheck_.size());
  trigvsnvtx_reco_ = iBooker.book2D("trigvsnvtx_reco",
                                    "Reco: Trigger vs. # vertices;N_{vertices};Trigger",
                                    100,
                                    0,
                                    100,
                                    hltPathsToCheck_.size(),
                                    0,
                                    hltPathsToCheck_.size());
  trigvsnvtx_ = iBooker.book2D("trigvsnvtx",
                               "Trigger vs. # vertices;N_{vertices};Trigger",
                               100,
                               0,
                               100,
                               hltPathsToCheck_.size(),
                               0,
                               hltPathsToCheck_.size());
}

void HigPhotonJetHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Count total number of events in one run
  evtsrun_++;

  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if (!triggerResults.isValid()) {
    edm::LogError("HigPhotonJetHLT") << "Missing triggerResults collection" << std::endl;
    return;
  }

  // Check whether contains monitored trigger and accepted
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
  bool triggered = isMonitoredTriggerAccepted(triggerNames, triggerResults);

  // if (!triggered) return;

  // Test scale
  // if (evtsrun_ > 10) return;

  // N Vertices
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(pvToken_, vertices);
  if (!vertices.isValid())
    return;
  if (verbose_)
    std::cout << "xshi:: N vertices : " << vertices->size() << std::endl;

  // Set trigger name labels
  for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
    triggers_->setBinLabel(i + 1, hltPathsToCheck_[i]);
  }

  // Fill trigger info
  for (unsigned int itrig = 0; itrig < triggerResults->size(); itrig++) {
    const std::string& triggername = triggerNames.triggerName(itrig);
    for (size_t i = 0; i < hltPathsToCheck_.size(); i++) {
      if (triggername.find(hltPathsToCheck_[i]) != std::string::npos) {
        triggers_reco_->Fill(i);
        trigvsnvtx_reco_->Fill(vertices->size(), i);
        if (triggered)
          triggers_->Fill(i);
        if (triggered)
          trigvsnvtx_->Fill(vertices->size(), i);
      }
    }
  }

  nvertices_reco_->Fill(vertices->size());
  if (triggered)
    nvertices_->Fill(vertices->size());

  // PF MET
  edm::Handle<reco::PFMETCollection> pfmets;
  iEvent.getByToken(pfMetToken_, pfmets);
  if (!pfmets.isValid())
    return;
  const reco::PFMET pfmet = pfmets->front();
  pfmet_reco_->Fill(pfmet.et());
  if (triggered)
    pfmet_->Fill(pfmet.et());
  if (verbose_)
    std::cout << "xshi:: number of pfmets: " << pfmets->size() << std::endl;

  pfmetphi_reco_->Fill(pfmet.phi());
  if (triggered)
    pfmetphi_->Fill(pfmet.phi());

  // Photons
  edm::Handle<reco::PhotonCollection> photons;
  iEvent.getByToken(photonsToken_, photons);
  if (!photons.isValid())
    return;
  int nphotons = 0;
  for (auto const& phoIter : *photons) {
    if (phoIter.pt() < photonMinPt_)
      continue;
    nphotons++;
    photonpt_reco_->Fill(phoIter.pt());
    photonrapidity_reco_->Fill(phoIter.rapidity());
    if (triggered)
      photonpt_->Fill(phoIter.pt());
    if (triggered)
      photonrapidity_->Fill(phoIter.rapidity());
    double tmp_delphiphomet = fabs(deltaPhi(phoIter.phi(), pfmet.phi()));
    delphiphomet_reco_->Fill(tmp_delphiphomet);
    if (triggered)
      delphiphomet_->Fill(tmp_delphiphomet);
  }
  nphotons_reco_->Fill(nphotons);
  if (triggered)
    nphotons_->Fill(nphotons);

  // PF Jet
  edm::Handle<reco::PFJetCollection> pfjets;
  iEvent.getByToken(pfJetsToken_, pfjets);
  if (!pfjets.isValid())
    return;
  if (verbose_)
    std::cout << "xshi:: N pfjets : " << pfjets->size() << std::endl;

  double min_delphijetmet = 6.0;
  TLorentzVector p4jet1, p4jet2, p4jj;
  // Two leading jets eta
  double etajet1(0), etajet2(0);
  int njet = 0;
  for (auto const& jetIter : *pfjets) {
    if (jetIter.pt() < pfjetMinPt_)
      continue;
    njet++;

    double tmp_delphijetmet = fabs(deltaPhi(jetIter.phi(), pfmet.phi()));
    if (tmp_delphijetmet < min_delphijetmet)
      min_delphijetmet = tmp_delphijetmet;

    if (njet == 1) {
      p4jet1.SetXYZM(jetIter.px(), jetIter.py(), jetIter.pz(), jetIter.mass());
      etajet1 = jetIter.eta();
    }
    if (njet == 2) {
      p4jet2.SetXYZM(jetIter.px(), jetIter.py(), jetIter.pz(), jetIter.mass());
      etajet2 = jetIter.eta();
    }
  }
  npfjets_reco_->Fill(njet);
  if (triggered)
    npfjets_->Fill(njet);

  delphijetmet_reco_->Fill(min_delphijetmet);
  if (triggered)
    delphijetmet_->Fill(min_delphijetmet);
  p4jj = p4jet1 + p4jet2;
  double deletajj = etajet1 - etajet2;
  if (verbose_)
    std::cout << "xshi:: invmass jj " << p4jj.M() << std::endl;

  invmassjj_reco_->Fill(p4jj.M());
  deletajj_reco_->Fill(deletajj);
  if (triggered)
    invmassjj_->Fill(p4jj.M());
  if (triggered)
    deletajj_->Fill(deletajj);
}

void HigPhotonJetHLTOfflineSource::dqmEndRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // Normalize to the total number of events in the run
  TH2F* h = trigvsnvtx_->getTH2F();
  double integral = h->Integral();
  double norm = (integral > 0.) ? evtsrun_ * hltPathsToCheck_.size() / integral : 1.;
  h->Scale(norm);
  if (verbose_) {
    std::cout << "xshi:: endRun total number of events: " << evtsrun_ << ", integral = " << h->Integral()
              << ", norm = " << norm << std::endl;
  }
}

bool HigPhotonJetHLTOfflineSource::isMonitoredTriggerAccepted(const edm::TriggerNames& triggerNames,
                                                              const edm::Handle<edm::TriggerResults>& triggerResults) {
  for (unsigned int itrig = 0; itrig < triggerResults->size(); itrig++) {
    // Only consider the triggered case.
    if (triggerAccept_ && ((*triggerResults)[itrig].accept() != 1))
      continue;
    const std::string& triggername = triggerNames.triggerName(itrig);
    for (auto const& i : hltPathsToCheck_) {
      if (triggername.find(i) != std::string::npos) {
        return true;
      }
    }
  }

  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HigPhotonJetHLTOfflineSource);
