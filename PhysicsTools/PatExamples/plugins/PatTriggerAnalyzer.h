#ifndef PhysicsTools_PatExamples_PatTriggerAnalyzer_h
#define PhysicsTools_PatExamples_PatTriggerAnalyzer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <map>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/PatCandidates/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "TH1D.h"
#include "TH2D.h"

namespace pat {

  class PatTriggerAnalyzer : public edm::EDAnalyzer {

    public:

      explicit PatTriggerAnalyzer( const edm::ParameterSet & iConfig );
      ~PatTriggerAnalyzer();

    private:

      virtual void beginJob() ;
      virtual void analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup );
      virtual void endJob();

      edm::InputTag trigger_;
      edm::InputTag triggerEvent_;
      edm::InputTag muons_;
      std::string   muonMatch_;

      std::map< std::string, TH1D* > histos1D_;
      std::map< std::string, TH2D* > histos2D_;
  };

}

#endif
