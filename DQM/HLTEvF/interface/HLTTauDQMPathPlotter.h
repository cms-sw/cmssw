/* HLTTau Path  Analyzer
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


class HLTTauDQMPathPlotter  {
  
 public:
   HLTTauDQMPathPlotter(const edm::ParameterSet&,bool);
  ~HLTTauDQMPathPlotter();
  void analyze(const edm::Event&, const edm::EventSetup&, const std::vector<LVColl>&);
  
 private:
  void endJob() ;
  LVColl getFilterCollection(size_t,int,const trigger::TriggerEventWithRefs&);



  //helper functions
  bool match(const LV&,const LVColl&,double);

  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag triggerEventObject_;

  //Just a tag for better file organization
  std::string triggerTag_;

  //The  filters
  std::vector<edm::InputTag> filter_;
  std::vector<int> TauType_;
  std::vector<int> LeptonType_;

  //Parameters(Note that the first entry is for the reference events)
  std::vector<unsigned> nTriggeredTaus_;
  std::vector<unsigned> nTriggeredLeptons_;

  bool doRefAnalysis_;
  std::vector<double> matchDeltaR_;

  //MonitorElements
  MonitorElement *accepted_events;
  MonitorElement *accepted_events_matched;




};
#endif
