/* HLTJetMET Path Validation Analyzer
Jochen Cammin
University of Rochester
cammin@fnal.gov

Extensions from Len Apanasevich.
*/


#ifndef HLTJetMETValidation_h
#define HLTJetMETValidation_h


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNames.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

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
  //JL    
  virtual void endRun(const edm::Run&, const edm::EventSetup&) ;

  void getHLTResults(const edm::TriggerResults&);

//JoCa  //helper functions
//JoCa  bool match(const LV&,const LVColl&,double);
//JoCa  std::vector<double> calcEfficiency(int,int);
//JoCa  
//JoCa
//JoCa
  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag triggerEventObject_;
  edm::InputTag CaloJetAlgorithm, GenJetAlgorithm, CaloMETColl, GenMETColl, HLTriggerResults;
//JoCa
//JoCa  //reference Collection
//JoCa  edm::InputTag refCollection_;
//JoCa  edm::InputTag refLeptonCollection_;
//JoCa    
  //Just a tag for better file organization
  std::string triggerTag_, MyTrigger;
//JoCa
  edm::InputTag _reffilter;
  edm::InputTag _probefilter;
  edm::InputTag _HLTPath;
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
  MonitorElement *_meRecoJetEta, *_meRecoJetEtaRef, *_meRecoJetEtaProbe, *_meRecoJetEtaTrg;
  MonitorElement *_meRecoJetPhi, *_meRecoJetPhiRef, *_meRecoJetPhiProbe, *_meRecoJetPhiTrg;
  MonitorElement *_meRecoJetPt, *_meRecoJetPtRef, *_meRecoJetPtProbe, *_meRecoJetPtTrg;
  MonitorElement *_meGenJetPt,  *_meGenJetPtRef , *_meGenJetPtProbe , *_meGenJetPtTrg;
  MonitorElement *_meGenJetEta,  *_meGenJetEtaRef , *_meGenJetEtaProbe , *_meGenJetEtaTrg;
  MonitorElement *_meGenJetPhi,  *_meGenJetPhiRef , *_meGenJetPhiProbe , *_meGenJetPhiTrg;
  MonitorElement *_meRecoMET,   *_meRecoMETRef  , *_meRecoMETProbe  , *_meRecoMETTrg;
  MonitorElement *_meGenMET,    *_meGenMETRef  ,  *_meGenMETProbe  ,  *_meGenMETTrg;
  MonitorElement *_meRefPt;
  MonitorElement *_meProbePt;
  MonitorElement *_triggerResults;

  //JL
  //MonitorElement *_meTurnOnMET;
  //MonitorElement *_meTurnOnJetPt;

//Define Numbers 

  int NRef;
  int NProbe;
  int evtCnt;

// store hlt information in a map
  std::vector<bool> hlttrigs;
  std::map <std::string,bool> hltTriggerMap;
  std::map<std::string,bool>::iterator trig_iter;

  edm::TriggerNames triggerNames_;  // TriggerNames class

  bool HLTinit_;

  //JL
  bool writeFile_;
};
#endif
