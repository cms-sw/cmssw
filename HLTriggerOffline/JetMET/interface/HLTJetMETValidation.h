/* HLTJetMET Path Validation Analyzer
   Migrated to use DQMEDAnalyzer by: Jyothsna Rani Komaragiri, Oct 2014
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
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <vector>
#include <string>
#include "TPRegexp.h"

namespace edm {
  class TriggerNames;
}

class HLTJetMETValidation : public DQMEDAnalyzer {
  
 public:
  explicit HLTJetMETValidation(const edm::ParameterSet&);
  ~HLTJetMETValidation();
  
 private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const & iRun, edm::EventSetup const & iSetup) override;
  virtual void dqmBeginRun(edm::Run const& iRun,edm::EventSetup const& iSetup) override;

  void getHLTResults(const edm::TriggerResults&,
                     const edm::TriggerNames & triggerNames);

  /// InputTag of TriggerEventWithRefs to analyze
  edm::EDGetTokenT<trigger::TriggerEventWithRefs> triggerEventObject_;
  edm::EDGetTokenT<reco::PFJetCollection> PFJetAlgorithm;
  edm::EDGetTokenT<reco::GenJetCollection> GenJetAlgorithm;
  edm::EDGetTokenT<reco::CaloMETCollection> CaloMETColl;
  edm::EDGetTokenT<reco::GenMETCollection> GenMETColl;
  edm::EDGetTokenT<edm::TriggerResults> HLTriggerResults;

  //Just a tag for better file organization
  std::string triggerTag_, patternJetTrg_, patternMetTrg_, patternMuTrg_;

  std::vector<MonitorElement*> _meHLTJetPt;
  std::vector<MonitorElement*> _meHLTJetPtTrgMC;
  std::vector<MonitorElement*> _meHLTJetPtTrg;
  std::vector<MonitorElement*> _meHLTJetPtTrgLow;
  std::vector<MonitorElement*> _meHLTJetEta;
  std::vector<MonitorElement*> _meHLTJetEtaTrgMC;
  std::vector<MonitorElement*> _meHLTJetEtaTrg;
  std::vector<MonitorElement*> _meHLTJetEtaTrgLow;
  std::vector<MonitorElement*> _meHLTJetPhi;
  std::vector<MonitorElement*> _meHLTJetPhiTrgMC;
  std::vector<MonitorElement*> _meHLTJetPhiTrg;
  std::vector<MonitorElement*> _meHLTJetPhiTrgLow;

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

  std::vector<MonitorElement*> _meHLTMET;
  std::vector<MonitorElement*> _meHLTMETTrgMC;
  std::vector<MonitorElement*> _meHLTMETTrg;
  std::vector<MonitorElement*> _meHLTMETTrgLow;  
  std::vector<MonitorElement*> _meGenMET;
  std::vector<MonitorElement*> _meGenMETTrgMC;
  std::vector<MonitorElement*> _meGenMETTrg;
  std::vector<MonitorElement*> _meGenMETTrgLow;  

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

  bool writeFile_;
};
#endif
