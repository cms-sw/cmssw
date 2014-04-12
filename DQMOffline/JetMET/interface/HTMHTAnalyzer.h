#ifndef HTMHTAnalyzer_H
#define HTMHTAnalyzer_H


/** \class HTMHTAnalyzer
 *
 *  DQM monitoring source for HTMHT
 *
 *  \author K. Hatakeyama, Rockefeller University
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class HTMHTAnalyzer : public DQMEDAnalyzer {
 public:

  /// Constructor
  HTMHTAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~HTMHTAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore * dbe);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
               const edm::TriggerResults&);

  int evtCounter;

 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  int _verbose;
  
  std::string metname;

  std::string _source;

  edm::InputTag theJetCollectionForHTMHTLabel;

  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;

  int _trig_JetMB;

  // Pt threshold for Jets
  double _ptThreshold;

  //the histos
  MonitorElement* jetME;

  MonitorElement* hNevents;

  MonitorElement* hNJets;

  MonitorElement* hMHx;
  MonitorElement* hMHy;
  MonitorElement* hMHT;
  MonitorElement* hMHTPhi;

  MonitorElement* hHT;

};
#endif
