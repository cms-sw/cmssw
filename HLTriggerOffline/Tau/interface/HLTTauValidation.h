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
  bool match(const reco::Candidate&,const LVColl&,double);
  double* calcEfficiency(int,int);
  


  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag triggerEventObject_;

  //reference Collection
  edm::InputTag refCollection_;
    
  //Just a tag for better file organization
  std::string triggerTag_;
  


  //The four basic filters
  std::string l1seedFilter_;
  std::string l2filter_;
  std::string l25filter_;
  std::string l3filter_;

  


  //Parameters
  int nTriggeredTaus_;
  bool doRefAnalysis_;
  std::string outFile_;
  std::string logFile_;
  double matchDeltaRL1_;
  double matchDeltaRHLT_;

  //MonitorElements

  /*Trigger Bits for Tau and Reference(electron) Trigger*/
  MonitorElement *triggerBits_Tau;
  MonitorElement *triggerBits_Ref;
  MonitorElement *etVsTriggerPath_Tau;
  MonitorElement *etVsTriggerPath_Ref;






  //Define Numbers 
  
  int NRefEvents;
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
