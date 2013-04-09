#ifndef PhysicsTools_PatExamples_PatTriggerAnalyzer_h
#define PhysicsTools_PatExamples_PatTriggerAnalyzer_h

#include <map>
#include <string>

#include "TH1D.h"
#include "TH2D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerEvent.h"


class PatTriggerAnalyzer : public edm::EDAnalyzer {
  
 public:
  /// default constructor  
  explicit PatTriggerAnalyzer( const edm::ParameterSet & iConfig );
  /// default destructor
  ~PatTriggerAnalyzer();
  
 private:
  /// everythin that needs to be done before the event loop
  virtual void beginJob();
  /// everythin that needs to be done during the event loop
  virtual void analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup );
  /// everythin that needs to be done after the event loop
  virtual void endJob();

  /// input for patTrigger  
  edm::InputTag trigger_;
  /// input for patTriggerEvent
  edm::InputTag triggerEvent_;
  /// input for muons
  edm::InputTag muons_;
  /// input for trigger match objects
  std::string   muonMatch_;
  /// minimal id for meanPt plot
  unsigned minID_;
  /// maximal id for meanPt plot
  unsigned maxID_;
  
  /// histogram management
  std::map< std::string, TH1D* > histos1D_;
  std::map< std::string, TH2D* > histos2D_;
  
  /// internals for meanPt histogram calculation
  std::map< unsigned, unsigned > sumN_;
  std::map< unsigned, double >   sumPt_;
};

#endif
