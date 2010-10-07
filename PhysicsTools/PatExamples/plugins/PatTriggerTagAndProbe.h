#ifndef PhysicsTools_PatExamples_PatTriggerTagAndProbe_h
#define PhysicsTools_PatExamples_PatTriggerTagAndProbe_h

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


class PatTriggerTagAndProbe : public edm::EDAnalyzer {
  
 public:
  /// default constructor
  explicit PatTriggerTagAndProbe( const edm::ParameterSet & iConfig );
  /// default destructor
  ~PatTriggerTagAndProbe();
  
 private:
  /// everythin that needs to be done before the event loop
  virtual void beginJob() ;
  /// everythin that needs to be done during the event loop
  virtual void analyze( const edm::Event & iEvent, const edm::EventSetup & iSetup );
  /// everythin that needs to be done after the event loop
  virtual void endJob();

  /// helper function to set proper bin errors
  void setErrors(TH1D& h, const TH1D& ref);
  
  /// input for patTrigger
  edm::InputTag trigger_;
  /// input for patTriggerEvent
  edm::InputTag triggerEvent_;
  /// input for muons
  edm::InputTag muons_;
  /// input for trigger match objects
  std::string   muonMatch_;
  
  /// management of 1d histograms
  std::map< std::string, TH1D* > histos1D_;
};

#endif
