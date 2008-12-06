#ifndef HLTriggerOffline_Tau_HLTTauElDQMOfflineSource_H
#define HLTriggerOffline_Tau_HLTTauElDQMOfflineSource_H


// Base Class Headers
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include "TDirectory.h"
#include "TH1F.h"
#include "TH2F.h"


// DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


typedef math::XYZTLorentzVectorD   LV;
typedef std::vector<LV>            LVColl;


class HLTTauElDQMOfflineSource : public edm::EDAnalyzer{
public:
  /// Constructor
  explicit HLTTauElDQMOfflineSource(const edm::ParameterSet& pset);

  /// Destructor
  ~HLTTauElDQMOfflineSource();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&);
  void endJob();

private:
  
  
  
  //edm::InputTag m_theL1Seed;
  edm::InputTag refCollection_;
  std::vector<int> m_theHLTOutputTypes;
  std::vector<bool> m_plotiso;
  std::vector<std::pair<double,double> > m_plotBounds; 
  std::vector<edm::InputTag> m_theHLTCollectionLabels; 
  std::vector<std::vector<edm::InputTag> > m_isoNames; // there has to be a better solution
  //std::string m_theHltName;
  
  unsigned int reqNum_;
  int   pdgGen_;
  double genEtaAcc_;
  double genEtAcc_;
  std::string outputFile_;
  std::string triggerName_;
  double thePtMin_ ;
  double thePtMax_ ;
  unsigned int theNbins_ ;
  
  std::vector<MonitorElement*> m_etahist;
  std::vector<MonitorElement*> m_ethist;
  std::vector<MonitorElement*> m_etahistmatch;
  std::vector<MonitorElement*> m_ethistmatch;
  MonitorElement* m_total;
  MonitorElement* m_etgen;
  MonitorElement* m_etagen;
  

  template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& ,const edm::Event& ,unsigned int,LVColl& );
  

};
#endif
