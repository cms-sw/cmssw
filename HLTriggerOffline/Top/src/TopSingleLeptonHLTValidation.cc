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
  iEvent.getByToken(tokElectrons_,electrons);
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
  iEvent.getByToken(tokMuons_,muons);
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
  iEvent.getByToken(tokJets_,jets);
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
  iEvent.getByToken(tokTrigger_,triggerTable);
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerTable);
  for (unsigned int i=0; i<triggerNames.triggerNames().size(); ++i) {

    TString name = triggerNames.triggerNames()[i].c_str();
    bool isInteresting = false;
    for (unsigned int j=0; j<vsPaths_.size(); j++) {
          if (name.Contains(TString(vsPaths_[j]), TString::kIgnoreCase)) isInteresting = true; 
        }

    if (isAll_ && isInteresting) isSel_ = true;

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
}


// ------------ method called once each job just before starting event loop  ------------
  void 
TopSingleLeptonHLTValidation::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
  void 
TopSingleLeptonHLTValidation::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
   void 
   TopSingleLeptonHLTValidation::beginRun(edm::Run const&, edm::EventSetup const&)
   {
   }
   */

// ------------ method called when ending the processing of a run  ------------

  void 
TopSingleLeptonHLTValidation::endRun(edm::Run const&, edm::EventSetup const&)
{
  dbe_->setCurrentFolder(sDir_);
  hEffLeptonPt  = dbe_->book1D("EfficiencyVsPtLepton", "EfficiencyVsPtLepton", 50, 0., 250.);
  hEffLeptonEta = dbe_->book1D("EfficiencyVsEtaLepton", "EfficiencyVsEtaLepton", 30, -3. , 3.);
  hEffJetPt     = dbe_->book1D("EfficiencyVsPtLastJet", "EfficiencyVsPtLastJet", 60, 0., 300.);
  hEffJetEta    = dbe_->book1D("EfficiencyVsEtaLastJet", "EfficiencyVsEtaLastJet", 30, -3., 3.);

  //------ Efficiency wrt
  // lepton pt
  for (int iBin = 1; iBin <= hNumLeptonPt->GetNbinsX(); ++iBin)
  {
    if(hDenLeptonPt->GetBinContent(iBin) == 0)
      hEffLeptonPt->setBinContent(iBin, 0.);
    else
      hEffLeptonPt->setBinContent(iBin, hNumLeptonPt->GetBinContent(iBin) / hDenLeptonPt->GetBinContent(iBin));
  }
  // lepton eta
  for (int iBin = 1; iBin <= hNumLeptonEta->GetNbinsX(); ++iBin)
  {
    if(hDenLeptonEta->GetBinContent(iBin) == 0)
      hEffLeptonEta->setBinContent(iBin, 0.);
    else
      hEffLeptonEta->setBinContent(iBin, hNumLeptonEta->GetBinContent(iBin) / hDenLeptonEta->GetBinContent(iBin));
  }
  // jet pt
  for (int iBin = 1; iBin <= hNumJetPt->GetNbinsX(); ++iBin)
  {
    if(hDenJetPt->GetBinContent(iBin) == 0)
      hEffJetPt->setBinContent(iBin, 0.);
    else
      hEffJetPt->setBinContent(iBin, hNumJetPt->GetBinContent(iBin) / hDenJetPt->GetBinContent(iBin));
  }
  // jet eta
  for (int iBin = 1; iBin <= hNumJetEta->GetNbinsX(); ++iBin)
  {
    if(hDenJetEta->GetBinContent(iBin) == 0)
      hEffJetEta->setBinContent(iBin, 0.);
    else
      hEffJetEta->setBinContent(iBin, hNumJetEta->GetBinContent(iBin) / hDenJetEta->GetBinContent(iBin));
  }
}


// ------------ method called when starting to processes a luminosity block  ------------
/*
   void 
   TopSingleLeptonHLTValidation::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */

// ------------ method called when ending the processing of a luminosity block  ------------
/*
   void 
   TopSingleLeptonHLTValidation::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
TopSingleLeptonHLTValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

