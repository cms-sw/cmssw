// -*- C++ -*-
//
// Package:    HLTriggerOffline/B2G
// Class:      B2GDoubleLeptonHLTValidation
// 
/**\class B2GDoubleLeptonHLTValidation B2GDoubleLeptonHLTValidation.cc HLTriggerOffline/B2G/src/B2GDoubleLeptonHLTValidation.cc

Description: 

Description: compute efficiencies of trigger paths on offline reco selection with respect to subleading lepton pt and eta

Implementation:
harvesting
*/
//
// Original Author:  Clint Richardson (copy of B2GSingleLeptonHLTValidation)
//         Created:  Tue, 05 Apr 2016 14:27:00 GMT
//
//
#include "HLTriggerOffline/B2G/interface/B2GDoubleLeptonHLTValidation.h"

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
B2GDoubleLeptonHLTValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  using namespace edm;

  isAll_ = false; isSel_ = false;

  // Electrons 
  Handle< edm::View<reco::GsfElectron> > electrons;
  if (!iEvent.getByToken(tokElectrons_,electrons))
    edm::LogWarning("B2GDoubleLeptonHLTValidation") << "Electrons collection not found \n";
  unsigned int nGoodE = 0;
  for(edm::View<reco::GsfElectron>::const_iterator e = electrons->begin(); e != electrons->end(); ++e){
    if (e->pt() < ptElectrons_) continue;
    if (fabs(e->eta()) > etaElectrons_) continue;
    nGoodE++;
    //leptons come sorted so use only 2nd
    if (nGoodE == 2) elec_ = electrons->ptrAt( e - electrons->begin());
  }
  // Muons 
  Handle< edm::View<reco::Muon> > muons;
  if (!iEvent.getByToken(tokMuons_,muons))
    edm::LogWarning("B2GDoubleLeptonHLTValidation") << "Muons collection not found \n";
  unsigned int nGoodM = 0;
  for(edm::View<reco::Muon>::const_iterator m = muons->begin(); m != muons->end(); ++m){
    if (!m->isPFMuon() || !m->isGlobalMuon()) continue;
    if (m->pt() < ptMuons_) continue;
    if (fabs(m->eta()) > etaMuons_) continue;
    nGoodM++;
    //leptons come sorted so use only 2nd
    if (nGoodM == 2) mu_ = muons->ptrAt( m - muons->begin() );
  }

  if (nGoodE >= minElectrons_ && nGoodM >= minMuons_ && nGoodE + nGoodM >= minLeptons_) isAll_ = true;


  //Trigger
  Handle<edm::TriggerResults> triggerTable;
  if (!iEvent.getByToken(tokTrigger_,triggerTable))
    edm::LogWarning("B2GDoubleLeptonHLTValidation") << "Trigger collection not found \n";
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerTable);
  bool isInteresting = false;
  for (unsigned int i=0; i<triggerNames.triggerNames().size(); ++i) {

    for (unsigned int j=0; j<vsPaths_.size(); j++) {
      if (triggerNames.triggerNames()[i].find(vsPaths_[j]) != std::string::npos) {
        isInteresting = true; 
        break;
      }
    }
    if (isInteresting) break;
  }

  if (isAll_ && isInteresting) isSel_ = true;
  else isSel_ = false;

  //Histos
  if (isAll_) {
    //here and below, change to nGoodE/M instead of min since we are taking subleading
    if (nGoodE > 1 && elec_.isNonnull() ) {
      hDenLeptonPt->Fill(elec_->pt());
      hDenLeptonEta->Fill(elec_->eta());
    }
    if (nGoodM > 1 && mu_.isNonnull() ) {
      hDenLeptonPt->Fill(mu_->pt());
      hDenLeptonEta->Fill(mu_->eta());
    }
    for(unsigned int idx=0; idx<vsPaths_.size(); ++idx){
      hDenTriggerMon->Fill(idx+0.5);
    }

  }
  if (isSel_) {
    if (nGoodE > 1 && elec_.isNonnull() ) {
      hNumLeptonPt->Fill(elec_->pt());
      hNumLeptonEta->Fill(elec_->eta());
    }
    if (nGoodM > 1 && mu_.isNonnull() ) {
      hNumLeptonPt->Fill(mu_->pt());
      hNumLeptonEta->Fill(mu_->eta());
    }

    for (unsigned int i=0; i<triggerNames.triggerNames().size(); ++i) {
      for (unsigned int j=0; j<vsPaths_.size(); j++) {
        if (triggerNames.triggerNames()[i].find(vsPaths_[j]) != std::string::npos) {
          hNumTriggerMon->Fill(j+0.5 );
        }
      }
    }
  }
}


// ------------ booking histograms -----------
  void
B2GDoubleLeptonHLTValidation::bookHistograms(DQMStore::IBooker & dbe, edm::Run const &, edm::EventSetup const &)
{
  dbe.setCurrentFolder(sDir_);
  hDenLeptonPt  = dbe.book1D("PtLeptonAll", "PtLeptonAll", 50, 0., 2500.);
  hDenLeptonEta = dbe.book1D("EtaLeptonAll", "EtaLeptonAll", 30, -3. , 3.);
  hNumLeptonPt  = dbe.book1D("PtLeptonSel", "PtLeptonSel", 50, 0., 2500.);
  hNumLeptonEta = dbe.book1D("EtaLeptonSel", "EtaLeptonSel", 30, -3. , 3.);
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
B2GDoubleLeptonHLTValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

