/* HLTTau Path Validation Analyzer
Michail Bachtis
University of Wisconsin - Madison
bachtis@hep.wisc.edu
*/


#ifndef HLTTauValidation_h
#define HLTTauValidation_h


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

typedef math::XYZTLorentzVectorD   LV;
typedef std::vector<LV>            LVColl;



class HLTTauValidation : public edm::EDAnalyzer {
  
 public:
  explicit HLTTauValidation(const edm::ParameterSet&);
  ~HLTTauValidation();
  
 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;


  //helper functions
  bool match(const LV&,const LVColl&,double);
  std::vector<double> calcEfficiency(int,int);
  


  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag triggerEventObject_;

  //reference Collection
  edm::InputTag refCollection_;
  edm::InputTag refLeptonCollection_;
    
  //Just a tag for better file organization
  std::string triggerTag_;

  //The four basic filters
  edm::InputTag l1seedFilter_;
  edm::InputTag l2filter_;
  edm::InputTag l25filter_;
  edm::InputTag l3filter_;


  //electron filter
  edm::InputTag electronFilter_;
  //muon filter
  edm::InputTag muonFilter_;


  //Parameters
  unsigned nTriggeredTaus_;
  unsigned nTriggeredLeptons_;
  bool doRefAnalysis_;
  std::string outFile_;
  std::string logFile_;
  double matchDeltaRL1_;
  double matchDeltaRHLT_;

  //MonitorElements

  /*Trigger Bits for Tau and Reference Trigger*/
  MonitorElement *l1eteff;
  MonitorElement *l2eteff;
  MonitorElement *l25eteff;
  MonitorElement *l3eteff;

  MonitorElement *refEt;
  MonitorElement *refEta;

  MonitorElement *l1etaeff;
  MonitorElement *l2etaeff;
  MonitorElement *l25etaeff;
  MonitorElement *l3etaeff;

  MonitorElement *accepted_events;
  MonitorElement *accepted_events_matched;



 

  //Define Numbers 
  int NRefEvents;
  int NLeptonEvents;
  int NLeptonEvents_Matched;
  int NL1Events;
  int NL1Events_Matched;
  int NL2Events;
  int NL2Events_Matched;
  int NL25Events;
  int NL25Events_Matched;
  int NL3Events;
  int NL3Events_Matched;
 

};
#endif
