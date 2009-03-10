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
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <vector>
#include "TDirectory.h"
#include "HepMC/GenParticle.h"

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
  std::vector<bool> plotiso;
  std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
  std::vector<std::pair<double,double> > plotBounds; 
  std::string theHltName;
  unsigned int reqNum;
  double thePtMin ;
  double thePtMax ;
  unsigned int theNbins ;

  std::vector<MonitorElement*> etahist;
  std::vector<MonitorElement*> ethist;
  std::vector<MonitorElement*> etahistmatch;
  std::vector<MonitorElement*> ethistmatch;
  std::vector<MonitorElement*> etahistiso;
  std::vector<MonitorElement*> ethistiso;
  MonitorElement* total;
  MonitorElement* etgen;
  MonitorElement* etagen;
  int   pdgGen;
  double genEtaAcc;
  double genEtAcc;

  template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& ,const edm::Event& ,unsigned int, std::vector<HepMC::GenParticle>& );
  
  DQMStore * dbe;
  std::string dirname_;

  edm::InputTag gencutCollection_;
  int gencut_;

};
#endif
