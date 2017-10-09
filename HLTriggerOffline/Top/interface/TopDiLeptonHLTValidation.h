// -*- C++ -*-
//
// Package:    HLTriggerOffline/Top
// Class:      TopDiLeptonHLTValidation
// 
/**\class TopDiLeptonHLTValidation TopDiLeptonHLTValidation.h HLTriggerOffline/Top/TopDiLeptonHLTValidation.h

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
#ifndef TOPDILEPTONHLTVALIDATION
#define TOPDILEPTONHLTVALIDATION

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"

//
// class declaration
//

class TopDiLeptonHLTValidation : public DQMEDAnalyzer {
   public:
      explicit TopDiLeptonHLTValidation(const edm::ParameterSet&);
      ~TopDiLeptonHLTValidation();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
      /// deduce monitorPath from label, the label is expected
      /// to be of type 'selectionPath:monitorPath'
      std::string monitorPath(const std::string& label) const { return label.substr(label.find(':')+1); };  
      /// set configurable labels for trigger monitoring histograms
      void triggerBinLabels(const std::vector<std::string>& labels);

      // ----------member data ---------------------------
      // DQM
      std::string sDir_;
      MonitorElement* hNumLeptonPt;
      MonitorElement* hDenLeptonPt;
      MonitorElement* hNumLeptonEta;
      MonitorElement* hDenLeptonEta;
      MonitorElement* hNumJetPt;
      MonitorElement* hDenJetPt;
      MonitorElement* hNumJetEta;
      MonitorElement* hDenJetEta;
      MonitorElement* hNumTriggerMon;
      MonitorElement* hDenTriggerMon;
      // Electrons
      const reco::GsfElectron *elec1_;
      const reco::GsfElectron *elec2_;
      std::string sElectrons_;
      edm::EDGetTokenT< edm::View<reco::GsfElectron> > tokElectrons_;
      double ptElectrons_;
      double etaElectrons_;
      double isoElectrons_;
      unsigned int minElectrons_;
      // Muons
      const reco::Muon *mu1_;
      const reco::Muon *mu2_;
      std::string sMuons_;
      edm::EDGetTokenT< edm::View<reco::Muon> > tokMuons_;
      double ptMuons_;
      double etaMuons_;
      double isoMuons_;
      unsigned int minMuons_;
      // Jets
      const reco::Jet *jet_;
      std::string sJets_;
      edm::EDGetTokenT< edm::View<reco::Jet> > tokJets_;
      double ptJets_;
      double etaJets_;
      unsigned int minJets_;
      // Trigger
      edm::InputTag iTrigger_;
      edm::EDGetTokenT<edm::TriggerResults> tokTrigger_;
      std::vector<std::string> vsPaths_;
      // Flags
      bool isAll_ = false;
      bool isSel_ = false;
};

inline void TopDiLeptonHLTValidation::triggerBinLabels(const std::vector<std::string>& labels)
{
  for(unsigned int idx=0; idx<labels.size(); ++idx){
    hNumTriggerMon->setBinLabel( idx+1, "["+monitorPath(labels[idx])+"]", 1);
    hDenTriggerMon->setBinLabel( idx+1, "["+monitorPath(labels[idx])+"]", 1);
  }
}

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TopDiLeptonHLTValidation::TopDiLeptonHLTValidation(const edm::ParameterSet& iConfig) :
  sDir_(iConfig.getUntrackedParameter<std::string>("sDir","Validation/Top/Efficiencies/")),
  sElectrons_(iConfig.getUntrackedParameter<std::string>("sElectrons","gsfElectrons")),
  ptElectrons_(iConfig.getUntrackedParameter<double>("ptElectrons",0.)),
  etaElectrons_(iConfig.getUntrackedParameter<double>("etaElectrons",0.)),
  isoElectrons_(iConfig.getUntrackedParameter<double>("isoElectrons",0.)),
  minElectrons_(iConfig.getUntrackedParameter<unsigned int>("minElectrons",0)),
  sMuons_(iConfig.getUntrackedParameter<std::string>("sMuons","muons")),
  ptMuons_(iConfig.getUntrackedParameter<double>("ptMuons",0.)),
  etaMuons_(iConfig.getUntrackedParameter<double>("etaMuons",0.)),
  isoMuons_(iConfig.getUntrackedParameter<double>("isoMuons",0.)),
  minMuons_(iConfig.getUntrackedParameter<unsigned int>("minMuons",0)),
  sJets_(iConfig.getUntrackedParameter<std::string>("sJets","ak5PFJets")),
  ptJets_(iConfig.getUntrackedParameter<double>("ptJets",0.)),
  etaJets_(iConfig.getUntrackedParameter<double>("etaJets",0.)),
  minJets_(iConfig.getUntrackedParameter<unsigned int>("minJets",0)),
  iTrigger_(iConfig.getUntrackedParameter<edm::InputTag>("iTrigger")),
  vsPaths_(iConfig.getUntrackedParameter< std::vector<std::string> >("vsPaths"))

{
  // Electrons
  tokElectrons_ = consumes< edm::View<reco::GsfElectron> >(edm::InputTag(sElectrons_));
  // Muons
  tokMuons_ = consumes< edm::View<reco::Muon> >(edm::InputTag(sMuons_));
  // Jets
  tokJets_ = consumes< edm::View<reco::Jet> >(edm::InputTag(sJets_));
  // Trigger
  tokTrigger_ = consumes<edm::TriggerResults>(iTrigger_); 
}


TopDiLeptonHLTValidation::~TopDiLeptonHLTValidation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}
#endif

//define this as a plug-in
DEFINE_FWK_MODULE(TopDiLeptonHLTValidation);
