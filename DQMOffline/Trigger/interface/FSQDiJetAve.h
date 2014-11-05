#ifndef FSQDiJetAve_H
#define FSQDiJetAve_H


// -*- C++ -*-
//
// Package:    DQMOffline/FSQDiJetAve
// Class:      FSQDiJetAve
// 
/**\class FSQDiJetAve FSQDiJetAve.cc DQMOffline/FSQDiJetAve/plugins/FSQDiJetAve.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Tomasz Fruboes
//         Created:  Tue, 04 Nov 2014 11:36:27 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Common/interface/TriggerNames.h"
//
// class declaration
//

namespace FSQ {
    class BaseHandler;
}

class FSQDiJetAve : public edm::EDAnalyzer {
   public:
      explicit FSQDiJetAve(const edm::ParameterSet&);
      ~FSQDiJetAve();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      bool m_isSetup;
      bool m_useGenWeight;
      DQMStore * m_dbe;
      std::string m_dirname;
      std::map<std::string,  MonitorElement*> m_me;

      edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken;
      edm::EDGetTokenT <edm::TriggerResults> triggerResultsFUToken;
      edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken;
      edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryFUToken;

      std::string processname_;
      edm::TriggerNames triggerNames_; // TriggerNames class
      edm::Handle<edm::TriggerResults> triggerResults_;
      edm::Handle<trigger::TriggerEvent> triggerObj_;
      edm::InputTag triggerSummaryLabel_;
      edm::InputTag triggerResultsLabel_;


      // TODO: auto ptr
      std::vector<FSQ::BaseHandler *> m_handlers;
};

#endif

