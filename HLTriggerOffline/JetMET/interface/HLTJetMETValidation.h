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
#include "FWCore/MessageLogger/interface/MessageLogger.h"


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
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <vector>
#include <string>
#include "TPRegexp.h"


namespace edm {
  class TriggerNames;
}

class HLTJetMETValidation : public edm::EDAnalyzer {
  
 public:
  explicit HLTJetMETValidation(const edm::ParameterSet&);
  ~HLTJetMETValidation();
  
 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  void getHLTResults(const edm::TriggerResults&,
                     const edm::TriggerNames & triggerNames);

  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag triggerEventObject_;
  edm::InputTag CaloJetAlgorithm, GenJetAlgorithm, CaloMETColl, GenMETColl, HLTriggerResults;

  //Just a tag for better file organization
  std::string triggerTag_, MyTrigger, patternJetTrg_, patternMetTrg_, patternMuTrg_;

  //edm::InputTag _HLTPath;
  //edm::InputTag _HLTLow;

  std::string outFile_;

  DQMStore* store;

  std::vector<MonitorElement*> _meRecoJetPt;
  std::vector<MonitorElement*> _meRecoJetPtTrgMC;
  std::vector<MonitorElement*> _meRecoJetPtTrg;
  std::vector<MonitorElement*> _meRecoJetPtTrgLow;
  std::vector<MonitorElement*> _meRecoJetEta;
  std::vector<MonitorElement*> _meRecoJetEtaTrgMC;
  std::vector<MonitorElement*> _meRecoJetEtaTrg;
  std::vector<MonitorElement*> _meRecoJetEtaTrgLow;
  std::vector<MonitorElement*> _meRecoJetPhi;
  std::vector<MonitorElement*> _meRecoJetPhiTrgMC;
  std::vector<MonitorElement*> _meRecoJetPhiTrg;
  std::vector<MonitorElement*> _meRecoJetPhiTrgLow;

  std::vector<MonitorElement*> _meGenJetPt;
  std::vector<MonitorElement*> _meGenJetPtTrgMC;
  std::vector<MonitorElement*> _meGenJetPtTrg;
  std::vector<MonitorElement*> _meGenJetPtTrgLow;
  std::vector<MonitorElement*> _meGenJetEta;
  std::vector<MonitorElement*> _meGenJetEtaTrgMC;
  std::vector<MonitorElement*> _meGenJetEtaTrg;
  std::vector<MonitorElement*> _meGenJetEtaTrgLow;
  std::vector<MonitorElement*> _meGenJetPhi;
  std::vector<MonitorElement*> _meGenJetPhiTrgMC;
  std::vector<MonitorElement*> _meGenJetPhiTrg;
  std::vector<MonitorElement*> _meGenJetPhiTrgLow;

  std::vector<MonitorElement*> _meRecoMET;
  std::vector<MonitorElement*> _meRecoMETTrgMC;
  std::vector<MonitorElement*> _meRecoMETTrg;
  std::vector<MonitorElement*> _meRecoMETTrgLow;  
  std::vector<MonitorElement*> _meGenMET;
  std::vector<MonitorElement*> _meGenMETTrgMC;
  std::vector<MonitorElement*> _meGenMETTrg;
  std::vector<MonitorElement*> _meGenMETTrgLow;  

  //MonitorElement *_meGenHT, *_meGenHTTrg, *_meGenHTTrgLow;
  //MonitorElement *_meRecoHT, *_meRecoHTTrg, *_meRecoHTTrgLow;
  MonitorElement *_triggerResults;

//Define Numbers 

  int evtCnt;

  HLTConfigProvider hltConfig_;
  std::vector<std::string> hltTrgJet;
  std::vector<std::string> hltTrgJetLow;
  std::vector<std::string> hltTrgMet;
  std::vector<std::string> hltTrgMetLow;

// store hlt information in a map
  std::vector<bool> hlttrigs;
  std::map <std::string,bool> hltTriggerMap;
  std::map<std::string,bool>::iterator trig_iter;

  bool HLTinit_;

  //JL
  bool writeFile_;
};
#endif
