#ifndef HLTriggerOffline_Egamma_EmDQM_H
#define HLTriggerOffline_Egamma_EmDQM_H


// Base Class Headers
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include <vector>
#include "TDirectory.h"
#include "TH1F.h"

class EmDQM : public edm::EDAnalyzer{
public:
  /// Constructor
  explicit EmDQM(const edm::ParameterSet& pset);

  /// Destructor
  ~EmDQM();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup&);
  void beginJob(const edm::EventSetup&);
  void endJob();

private:
  // Input from cfg file
  std::vector<edm::InputTag> theHLTCollectionLabels;  
  edm::InputTag theL1Seed;
  std::vector<int> theHLTOutputTypes;
  std::string theHltName;
  unsigned int reqNum;
  double thePtMin ;
  double thePtMax ;
  unsigned int theNbins ;

  std::vector<TH1F*> etahist;
  std::vector<TH1F*> ethist;
  TH1F* total;
  TH1F* etgen;
  TH1F* etagen;
  int   pdgGen;
  double genEtaAcc;
  double genEtAcc;

  template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& , std::vector<int>& ,int);
  

};
#endif
