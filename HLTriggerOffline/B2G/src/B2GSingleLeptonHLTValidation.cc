// -*- C++ -*-
//
// Package:    HLTriggerOffline/B2G
// Class:      B2GSingleLeptonHLTValidation
//
/**\class B2GSingleLeptonHLTValidation B2GSingleLeptonHLTValidation.cc
HLTriggerOffline/B2G/src/B2GSingleLeptonHLTValidation.cc

Description:

Description: compute efficiencies of trigger paths on offline reco selection
with respect to pt and eta

Implementation:
harvesting
*/
//
// Original Author:  Elvire Bouvier
//         Created:  Thu, 16 Jan 2014 16:27:35 GMT
//
//
#include "HLTriggerOffline/B2G/interface/B2GSingleLeptonHLTValidation.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "TString.h"
//
// member functions
//

// ------------ method called for each event  ------------
void B2GSingleLeptonHLTValidation::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  isAll_ = false;
  isSel_ = false;

  // Electrons
  Handle<edm::View<reco::GsfElectron>> electrons;
  if (!iEvent.getByToken(tokElectrons_, electrons))
    edm::LogWarning("B2GSingleLeptonHLTValidation") << "Electrons collection not found \n";
  unsigned int nGoodE = 0;
  for (edm::View<reco::GsfElectron>::const_iterator e = electrons->begin(); e != electrons->end(); ++e) {
    if (e->pt() < ptElectrons_)
      continue;
    if (fabs(e->eta()) > etaElectrons_)
      continue;
    nGoodE++;
    if (nGoodE == 1)
      elec_ = electrons->ptrAt(e - electrons->begin());
  }
  // Muons
  Handle<edm::View<reco::Muon>> muons;
  if (!iEvent.getByToken(tokMuons_, muons))
    edm::LogWarning("B2GSingleLeptonHLTValidation") << "Muons collection not found \n";
  unsigned int nGoodM = 0;
  for (edm::View<reco::Muon>::const_iterator m = muons->begin(); m != muons->end(); ++m) {
    if (!m->isPFMuon() || !m->isGlobalMuon())
      continue;
    if (m->pt() < ptMuons_)
      continue;
    if (fabs(m->eta()) > etaMuons_)
      continue;
    nGoodM++;
    if (nGoodM == 1)
      mu_ = muons->ptrAt(m - muons->begin());
  }
  // Jets
  Handle<edm::View<reco::Jet>> jets;
  if (!iEvent.getByToken(tokJets_, jets))
    edm::LogWarning("B2GSingleLeptonHLTValidation") << "Jets collection not found \n";
  unsigned int nGoodJ = 0;

  // Check to see if we want asymmetric jet pt cuts
  if (ptJets0_ > 0.0 || ptJets1_ > 0.0) {
    if (ptJets0_ > 0.0) {
      if (!jets->empty() && jets->at(0).pt() > ptJets0_) {
        nGoodJ++;
        jet_ = jets->ptrAt(0);
      }
    }
    if (ptJets1_ > 0.0) {
      if (jets->size() > 1 && jets->at(1).pt() > ptJets1_) {
        nGoodJ++;
        jet_ = jets->ptrAt(1);
      }
    }
  } else {
    for (edm::View<reco::Jet>::const_iterator j = jets->begin(); j != jets->end(); ++j) {
      if (j->pt() < ptJets_)
        continue;
      if (fabs(j->eta()) > etaJets_)
        continue;
      nGoodJ++;
      if (nGoodJ == minJets_)
        jet_ = jets->ptrAt(j - jets->begin());
    }
  }

  if (nGoodE >= minElectrons_ && nGoodM >= minMuons_ && nGoodJ >= minJets_)
    isAll_ = true;

  // Trigger
  Handle<edm::TriggerResults> triggerTable;
  if (!iEvent.getByToken(tokTrigger_, triggerTable))
    edm::LogWarning("B2GSingleLeptonHLTValidation") << "Trigger collection not found \n";
  const edm::TriggerNames &triggerNames = iEvent.triggerNames(*triggerTable);
  bool isInteresting = false;
  for (unsigned int i = 0; i < triggerNames.triggerNames().size(); ++i) {
    TString name = triggerNames.triggerNames()[i].c_str();
    for (unsigned int j = 0; j < vsPaths_.size(); j++) {
      if (name.Contains(TString(vsPaths_[j]), TString::kIgnoreCase)) {
        isInteresting = true;
        break;
      }
    }
    if (isInteresting)
      break;
  }

  if (isAll_ && isInteresting)
    isSel_ = true;
  else
    isSel_ = false;

  // Histos
  if (isAll_) {
    if (minElectrons_ > 0 && elec_.isNonnull()) {
      hDenLeptonPt->Fill(elec_->pt());
      hDenLeptonEta->Fill(elec_->eta());
    }
    if (minMuons_ > 0 && mu_.isNonnull()) {
      hDenLeptonPt->Fill(mu_->pt());
      hDenLeptonEta->Fill(mu_->eta());
    }
    if (jet_.isNonnull()) {
      hDenJetPt->Fill(jet_->pt());
      hDenJetEta->Fill(jet_->eta());
    }
    for (unsigned int idx = 0; idx < vsPaths_.size(); ++idx) {
      hDenTriggerMon->Fill(idx + 0.5);
    }
  }
  if (isSel_) {
    if (minElectrons_ > 0 && elec_.isNonnull()) {
      hNumLeptonPt->Fill(elec_->pt());
      hNumLeptonEta->Fill(elec_->eta());
    }
    if (minMuons_ > 0 && mu_.isNonnull()) {
      hNumLeptonPt->Fill(mu_->pt());
      hNumLeptonEta->Fill(mu_->eta());
    }
    if (jet_.isNonnull()) {
      hNumJetPt->Fill(jet_->pt());
      hNumJetEta->Fill(jet_->eta());
    }
    for (unsigned int i = 0; i < triggerNames.triggerNames().size(); ++i) {
      TString name = triggerNames.triggerNames()[i].c_str();
      for (unsigned int j = 0; j < vsPaths_.size(); j++) {
        if (name.Contains(TString(vsPaths_[j]), TString::kIgnoreCase)) {
          hNumTriggerMon->Fill(j + 0.5);
        }
      }
    }
  }
}

// ------------ booking histograms -----------
void B2GSingleLeptonHLTValidation::bookHistograms(DQMStore::IBooker &dbe, edm::Run const &, edm::EventSetup const &) {
  dbe.setCurrentFolder(sDir_);
  hDenLeptonPt = dbe.book1D("PtLeptonAll", "PtLeptonAll", 50, 0., 2500.);
  hDenLeptonEta = dbe.book1D("EtaLeptonAll", "EtaLeptonAll", 30, -3., 3.);
  hDenJetPt = dbe.book1D("PtLastJetAll", "PtLastJetAll", 60, 0., 3000.);
  hDenJetEta = dbe.book1D("EtaLastJetAll", "EtaLastJetAll", 30, -3., 3.);
  hNumLeptonPt = dbe.book1D("PtLeptonSel", "PtLeptonSel", 50, 0., 2500.);
  hNumLeptonEta = dbe.book1D("EtaLeptonSel", "EtaLeptonSel", 30, -3., 3.);
  hNumJetPt = dbe.book1D("PtLastJetSel", "PtLastJetSel", 60, 0., 3000.);
  hNumJetEta = dbe.book1D("EtaLastJetSel", "EtaLastJetSel", 30, -3., 3.);
  // determine number of bins for trigger monitoring
  unsigned int nPaths = vsPaths_.size();
  // monitored trigger occupancy for single lepton triggers
  hNumTriggerMon = dbe.book1D("TriggerMonSel", "TriggerMonSel", nPaths, 0., nPaths);
  hDenTriggerMon = dbe.book1D("TriggerMonAll", "TriggerMonAll", nPaths, 0., nPaths);
  // set bin labels for trigger monitoring
  triggerBinLabels(vsPaths_);
}

// ------------ method fills 'descriptions' with the allowed parameters for the
// module  ------------
void B2GSingleLeptonHLTValidation::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // The following says we do not know what parameters are allowed so do no
  // validation
  // Please change this to state exactly what you do use, even if it is no
  // parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
