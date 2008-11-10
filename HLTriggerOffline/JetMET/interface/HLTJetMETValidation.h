/* HLTJetMET Path Validation Analyzer
Michail Bachtis
University of Wisconsin - Madison
bachtis@hep.wisc.edu
Adapted for JetMET by Jochen Cammin <cammin@fnal.gov>
*/


#ifndef HLTJetMETValidation_h
#define HLTJetMETValidation_h


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



class HLTJetMETValidation : public edm::EDAnalyzer {
  
 public:
  explicit HLTJetMETValidation(const edm::ParameterSet&);
  ~HLTJetMETValidation();
  
 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;


//JoCa  //helper functions
//JoCa  bool match(const LV&,const LVColl&,double);
//JoCa  std::vector<double> calcEfficiency(int,int);
//JoCa  
//JoCa
//JoCa
//JoCa  /// InputTag of TriggerEventWithRefs to analyze
//JoCa  edm::InputTag triggerEventObject_;
//JoCa
//JoCa  //reference Collection
//JoCa  edm::InputTag refCollection_;
//JoCa  edm::InputTag refLeptonCollection_;
//JoCa    
  //Just a tag for better file organization
  std::string triggerTag_;
//JoCa
//JoCa  //The four basic filters
//JoCa  edm::InputTag l1seedFilter_;
//JoCa  edm::InputTag l2filter_;
//JoCa  edm::InputTag l25filter_;
//JoCa  edm::InputTag l3filter_;
//JoCa
//JoCa
//JoCa  //electron filter
//JoCa  edm::InputTag electronFilter_;
//JoCa  //muon filter
//JoCa  edm::InputTag muonFilter_;
//JoCa
//JoCa
//JoCa  //Parameters
//JoCa  unsigned nTriggeredTaus_;
//JoCa  unsigned nTriggeredLeptons_;
//JoCa  bool doRefAnalysis_;
  std::string outFile_;
//JoCa  std::string logFile_;
//JoCa  double matchDeltaRL1_;
//JoCa  double matchDeltaRHLT_;
//JoCa
//JoCa  //MonitorElements
//JoCa
//JoCa  /*Trigger Bits for Tau and Reference Trigger*/
  MonitorElement *test_histo;
//JoCa  MonitorElement *l1eteff;
//JoCa  MonitorElement *l2eteff;
//JoCa  MonitorElement *l25eteff;
//JoCa  MonitorElement *l3eteff;
//JoCa
//JoCa  MonitorElement *refEt;
//JoCa  MonitorElement *refEta;
//JoCa
//JoCa
//JoCa
//JoCa  MonitorElement *l1etaeff;
//JoCa  MonitorElement *l2etaeff;
//JoCa  MonitorElement *l25etaeff;
//JoCa  MonitorElement *l3etaeff;
//JoCa 
//JoCa
//JoCa  //Define Numbers 
//JoCa  int NRefEvents;
//JoCa  int NLeptonEvents;
//JoCa  int NLeptonEvents_Matched;
//JoCa  int NL1Events;
//JoCa  int NL1Events_Matched;
//JoCa  int NL2Events;
//JoCa  int NL2Events_Matched;
//JoCa  int NL25Events;
//JoCa  int NL25Events_Matched;
//JoCa  int NL3Events;
//JoCa  int NL3Events_Matched;
//JoCa 

};
#endif
