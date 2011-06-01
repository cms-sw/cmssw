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

  void getHLTResults(const edm::TriggerResults&,
                     const edm::TriggerNames & triggerNames);

  /// InputTag of TriggerEventWithRefs to analyze
  edm::InputTag triggerEventObject_;
  edm::InputTag CaloJetAlgorithm, GenJetAlgorithm, CaloMETColl, GenMETColl, HLTriggerResults;

  //Just a tag for better file organization
  std::string triggerTag_, MyTrigger;

  edm::InputTag _HLTPath;
  edm::InputTag _HLTLow;

  std::string outFile_;

  MonitorElement *_meRecoJetEta, *_meRecoJetEtaTrg, *_meRecoJetEtaTrgLow;
  MonitorElement *_meRecoJetPhi, *_meRecoJetPhiTrg, *_meRecoJetPhiTrgLow;
  MonitorElement *_meRecoJetPt, *_meRecoJetPtTrg, *_meRecoJetPtTrgLow;
  MonitorElement *_meGenJetPt, *_meGenJetPtTrg,  *_meGenJetPtTrgLow;
  MonitorElement *_meGenJetEta, *_meGenJetEtaTrg, *_meGenJetEtaTrgLow;
  MonitorElement *_meGenJetPhi, *_meGenJetPhiTrg, *_meGenJetPhiTrgLow;
  MonitorElement *_meRecoMET, *_meRecoMETTrg, *_meRecoMETTrgLow;
  MonitorElement *_meGenMET, *_meGenMETTrg, *_meGenMETTrgLow;
  MonitorElement *_meGenHT, *_meGenHTTrg, *_meGenHTTrgLow;
  MonitorElement *_meRecoHT, *_meRecoHTTrg, *_meRecoHTTrgLow;
  MonitorElement *_triggerResults;

//Define Numbers 

  int evtCnt;

// store hlt information in a map
  std::vector<bool> hlttrigs;
  std::map <std::string,bool> hltTriggerMap;
  std::map<std::string,bool>::iterator trig_iter;

  bool HLTinit_;

  //JL
  bool writeFile_;
};
#endif
