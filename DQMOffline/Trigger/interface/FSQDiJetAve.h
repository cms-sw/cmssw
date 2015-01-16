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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "HLTrigger/HLTcore/interface/TriggerExpressionData.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionEvaluator.h"
#include "HLTrigger/HLTcore/interface/TriggerExpressionParser.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

//
// class declaration
//

namespace FSQ {
    class BaseHandler;
}

class FSQDiJetAve : public DQMEDAnalyzer {
   public:
      explicit FSQDiJetAve(const edm::ParameterSet&);
      ~FSQDiJetAve();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

      virtual void bookHistograms(DQMStore::IBooker &, edm::Run const & run, edm::EventSetup const & c) override;
      virtual void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override;
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
      //
      triggerExpression::Data m_eventCache;
      bool m_useGenWeight;
      HLTConfigProvider m_hltConfig;


      std::string m_dirname;
      std::map<std::string,  MonitorElement*> m_me;

      edm::EDGetTokenT <edm::TriggerResults> triggerResultsToken;
      edm::EDGetTokenT <edm::TriggerResults> triggerResultsFUToken;
      edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken;
      edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryFUToken;
      edm::EDGetTokenT <GenEventInfoProduct  > m_genEvInfoToken;

      edm::TriggerNames m_triggerNames; // TriggerNames class
      edm::Handle<edm::TriggerResults> m_triggerResults;
      edm::Handle<trigger::TriggerEvent> m_trgEvent;
      edm::InputTag triggerSummaryLabel_;
      edm::InputTag triggerResultsLabel_;


      // TODO: auto ptr
      std::vector< std::shared_ptr<FSQ::BaseHandler> > m_handlers;
};

#endif

