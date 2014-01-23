// -*- C++ -*-
//
// Package:    HLTriggerOffline/Top
// Class:      TopSingleLeptonHLTValidation
// 
/**\class TopSingleLeptonHLTValidation TopSingleLeptonHLTValidation.h HLTriggerOffline/Top/interface/TopSingleLeptonHLTValidation.h

 Description: compute efficiencies of trigger paths on offline reco selection with respect to pt and eta

 Implementation:
     harvesting
*/
//
// Original Author:  Elvire Bouvier
//         Created:  Thu, 16 Jan 2014 16:27:35 GMT
//
//
#ifndef TOPSINGLELEPTONHLTVALIDATION
#define TOPSINGLELEPTONHLTVALIDATION

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

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
#include "TH1.h"

//
// class declaration
//

class TopSingleLeptonHLTValidation : public edm::EDAnalyzer {
   public:
      explicit TopSingleLeptonHLTValidation(const edm::ParameterSet&);
      ~TopSingleLeptonHLTValidation();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      // DQM
      DQMStore* dbe_;
      std::string sDir_;
      MonitorElement* hEffLeptonPt;
      MonitorElement* hEffLeptonEta;
      MonitorElement* hEffJetPt;
      MonitorElement* hEffJetEta;
      // Electrons
      const reco::GsfElectron *elec_;
      std::string sElectrons_;
      edm::EDGetTokenT< edm::View<reco::GsfElectron> > tokElectrons_;
      double ptElectrons_;
      double etaElectrons_;
      double isoElectrons_;
      unsigned int minElectrons_;
      // Muons
      const reco::Muon *mu_;
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
      std::string sTrigger_;
      edm::EDGetTokenT<edm::TriggerResults> tokTrigger_;
      std::vector<std::string> vsPaths_;
      // Histos
      bool isAll_ = false;
      TH1F *hDenLeptonPt  = new TH1F("PtLeptonAll", "PtLeptonAll", 50, 0., 250.);
      TH1F *hDenLeptonEta = new TH1F("EtaLeptonAll", "EtaLeptonAll", 30, -3. , 3.);
      TH1F *hDenJetPt     = new TH1F("PtLastJetAll", "PtLastJetAll", 60, 0., 300.);
      TH1F *hDenJetEta    = new TH1F("EtaLastJetAll", "EtaLastJetAll", 30, -3., 3.);
      bool isSel_ = false;
      TH1F *hNumLeptonPt  = new TH1F("PtLeptonSel", "PtLeptonSel", 50, 0., 250.);
      TH1F *hNumLeptonEta = new TH1F("EtaLeptonSel", "EtaLeptonSel", 30, -3. , 3.);
      TH1F *hNumJetPt     = new TH1F("PtLastJetSel", "PtLastJetSel", 60, 0., 300.);
      TH1F *hNumJetEta    = new TH1F("EtaLastJetSel", "EtaLastJetSel", 30, -3., 3.);
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TopSingleLeptonHLTValidation::TopSingleLeptonHLTValidation(const edm::ParameterSet& iConfig) :
  sDir_(iConfig.getUntrackedParameter<std::string>("sDir","HLTValidation/Top/Efficiencies/")),
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
  sTrigger_(iConfig.getUntrackedParameter<std::string>("sTrigger","TriggerResults")),
  vsPaths_(iConfig.getUntrackedParameter< std::vector<std::string> >("vsPaths"))

{
  dbe_ = edm::Service<DQMStore>().operator->();
  // Electrons
  tokElectrons_ = consumes< edm::View<reco::GsfElectron> >(edm::InputTag(sElectrons_));
  // Muons
  tokMuons_ = consumes< edm::View<reco::Muon> >(edm::InputTag(sMuons_));
  // Jets
  tokJets_ = consumes< edm::View<reco::Jet> >(edm::InputTag(sJets_));
  // Trigger
  tokTrigger_ = consumes<edm::TriggerResults>(edm::InputTag(sTrigger_));
}


TopSingleLeptonHLTValidation::~TopSingleLeptonHLTValidation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}
#endif

//define this as a plug-in
DEFINE_FWK_MODULE(TopSingleLeptonHLTValidation);
