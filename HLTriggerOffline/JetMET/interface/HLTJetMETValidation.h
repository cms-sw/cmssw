/* HLTJetMET Path Validation Analyzer
Jochen Cammin
University of Rochester
cammin@fnal.gov
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
  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag triggerEventObject_;
//JoCa
//JoCa  //reference Collection
//JoCa  edm::InputTag refCollection_;
//JoCa  edm::InputTag refLeptonCollection_;
//JoCa    
  //Just a tag for better file organization
  std::string triggerTag_;
//JoCa
  edm::InputTag _reffilter;
  edm::InputTag _probefilter;
//JoCa  //Parameters
  std::string outFile_;
//JoCa  std::string logFile_;
//JoCa  double matchDeltaRL1_;
//JoCa  double matchDeltaRHLT_;
//JoCa
//JoCa  //MonitorElements
//JoCa
//JoCa  /*Trigger Bits for Tau and Reference Trigger*/
  MonitorElement *test_histo;
  MonitorElement *_meSingleJetPt;
  MonitorElement *_meRefPt;
  MonitorElement *_meProbePt;

//Define Numbers 

  int NRef;
  int NProbe;

};
#endif
