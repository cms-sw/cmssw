// -*- C++ -*-
//
// Package:    HLTriggerOffline/Top
// Class:      TopSingleLeptonHLTValidation
// 
/**\class TopSingleLeptonHLTValidation TopSingleLeptonHLTValidation.cc HLTriggerOffline/Top/src/TopSingleLeptonHLTValidation.cc

Description: 

Description: compute efficiencies of trigger paths on offline reco selection with respect to pt and eta

Implementation:
harvesting
*/
//
// Original Author:  Elvire Bouvier
//         Created:  Thu, 16 Jan 2014 16:27:35 GMT
//
//
#include "HLTriggerOffline/Top/interface/TopSingleLeptonHLTValidation.h"

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "TString.h"
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
//
// member functions
//

// ------------ method called for each event  ------------
  void
TopSingleLeptonHLTValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  using namespace edm;

  isAll_ = false; isSel_ = false;

  // Electrons 
  Handle< edm::View<reco::GsfElectron> > electrons;
  if (!iEvent.getByToken(tokElectrons_,electrons))
    edm::LogWarning("TopSingleLeptonHLTValidation") << "Electrons collection not found \n";
  unsigned int nGoodE = 0;
  for(edm::View<reco::GsfElectron>::const_iterator e = electrons->begin(); e != electrons->end(); ++e){
    if (e->pt() < ptElectrons_) continue;
    if (fabs(e->eta()) > etaElectrons_) continue;
    if ((e->dr03TkSumPt()+e->dr03EcalRecHitSumEt()+e->dr03HcalTowerSumEt())/e->pt() > isoElectrons_ ) continue;
    nGoodE++;
    if (nGoodE == 1) elec_ = &(*e);
  }
  // Muons 
  Handle< edm::View<reco::Muon> > muons;
  if (!iEvent.getByToken(tokMuons_,muons))
    edm::LogWarning("TopSingleLeptonHLTValidation") << "Muons collection not found \n";
  unsigned int nGoodM = 0;
  for(edm::View<reco::Muon>::const_iterator m = muons->begin(); m != muons->end(); ++m){
    if (!m->isPFMuon() || !m->isGlobalMuon()) continue;
    if (m->pt() < ptMuons_) continue;
    if (fabs(m->eta()) > etaMuons_) continue;
    if (((m->pfIsolationR04()).sumChargedHadronPt+(m->pfIsolationR04()).sumPhotonEt+(m->pfIsolationR04()).sumNeutralHadronEt)/m->pt() > isoMuons_ ) continue;
    nGoodM++;
    if (nGoodM == 1) mu_ = &(*m);
  }
  // Jets 
  Handle< edm::View<reco::Jet> > jets;
  if (!iEvent.getByToken(tokJets_,jets)) 
    edm::LogWarning("TopSingleLeptonHLTValidation") << "Jets collection not found \n";
  unsigned int nGoodJ = 0;
  if (minJets_ == 4) {
    for(edm::View<reco::Jet>::const_iterator j = jets->begin(); j != jets->end(); ++j){
      if (nGoodJ < 1 && j->pt() < 55) continue;
      if (nGoodJ < 2 && j->pt() < 45) continue;
      if (nGoodJ < 3 && j->pt() < 35) continue;
      if (nGoodJ >= 3 && j->pt() < 20) continue;
      if (fabs(j->eta()) > etaJets_) continue;
      nGoodJ++;
      if (nGoodJ == minJets_) jet_ = &(*j);
    }
  }
  else {
    for(edm::View<reco::Jet>::const_iterator j = jets->begin(); j != jets->end(); ++j){
      if (j->pt() < ptJets_) continue;
      if (fabs(j->eta()) > etaJets_) continue;
      nGoodJ++;
      if (nGoodJ == minJets_) jet_ = &(*j);
    }
  }

  if (nGoodE >= minElectrons_ && nGoodM >= minMuons_ && nGoodJ >= minJets_) isAll_ = true;


  //Trigger
  Handle<edm::TriggerResults> triggerTable;
  if (!iEvent.getByToken(tokTrigger_,triggerTable))
    edm::LogWarning("TopSingleLeptonHLTValidation") << "Trigger collection not found \n";
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerTable);
  unsigned int isInteresting = 0;
  for (unsigned int i=0; i<triggerNames.triggerNames().size(); ++i) {

    TString name = triggerNames.triggerNames()[i].c_str();
    for (unsigned int j=0; j<vsPaths_.size(); j++) {
      if (name.Contains(TString(vsPaths_[j]), TString::kIgnoreCase)) {
        if (triggerTable->accept(i)){
          isInteresting++;   
          if (isAll_) hNumTriggerMon->Fill(j+0.5 ); 
        }
      }
    }
  }

  if (isAll_ && isInteresting > 0) isSel_ = true;
  else isSel_ = false;

  //Histos
  if (isAll_) {
    if (minElectrons_ > 0) {
      hDenLeptonPt->Fill(elec_->pt());
      hDenLeptonEta->Fill(elec_->eta());
    }
    if (minMuons_ > 0) {
      hDenLeptonPt->Fill(mu_->pt());
      hDenLeptonEta->Fill(mu_->eta());
    }
    hDenJetPt->Fill(jet_->pt());
    hDenJetEta->Fill(jet_->eta());
    for(unsigned int idx=0; idx<vsPaths_.size(); ++idx){
      hDenTriggerMon->Fill(idx+0.5);
    }

  }
  if (isSel_) {
    if (minElectrons_ > 0) {
      hNumLeptonPt->Fill(elec_->pt());
      hNumLeptonEta->Fill(elec_->eta());
    }
    if (minMuons_ > 0) {
      hNumLeptonPt->Fill(mu_->pt());
      hNumLeptonEta->Fill(mu_->eta());
    }
    hNumJetPt->Fill(jet_->pt());
    hNumJetEta->Fill(jet_->eta());
  }
}


// ------------ booking histograms -----------
  void
TopSingleLeptonHLTValidation::bookHistograms(DQMStore::IBooker & dbe, edm::Run const &, edm::EventSetup const &)
{
  dbe.setCurrentFolder(sDir_);
  hDenLeptonPt  = dbe.book1D("PtLeptonAll", "PtLeptonAll", 50, 0., 250.);
  hDenLeptonEta = dbe.book1D("EtaLeptonAll", "EtaLeptonAll", 30, -3. , 3.);
  hDenJetPt     = dbe.book1D("PtLastJetAll", "PtLastJetAll", 60, 0., 300.);
  hDenJetEta    = dbe.book1D("EtaLastJetAll", "EtaLastJetAll", 30, -3., 3.);
  hNumLeptonPt  = dbe.book1D("PtLeptonSel", "PtLeptonSel", 50, 0., 250.);
  hNumLeptonEta = dbe.book1D("EtaLeptonSel", "EtaLeptonSel", 30, -3. , 3.);
  hNumJetPt     = dbe.book1D("PtLastJetSel", "PtLastJetSel", 60, 0., 300.);
  hNumJetEta    = dbe.book1D("EtaLastJetSel", "EtaLastJetSel", 30, -3., 3.);
  // determine number of bins for trigger monitoring
  unsigned int nPaths = vsPaths_.size();
  // monitored trigger occupancy for single lepton triggers
  hNumTriggerMon    = dbe.book1D("TriggerMonSel" , "TriggerMonSel", nPaths, 0.,  nPaths);
  hDenTriggerMon    = dbe.book1D("TriggerMonAll" , "TriggerMonAll", nPaths, 0.,  nPaths);
  // set bin labels for trigger monitoring
  triggerBinLabels(vsPaths_);
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TopSingleLeptonHLTValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

