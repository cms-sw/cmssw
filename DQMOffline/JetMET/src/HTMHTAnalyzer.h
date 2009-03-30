#ifndef HTMHTAnalyzer_H
#define HTMHTAnalyzer_H


/** \class HTMHTAnalyzer
 *
 *  DQM monitoring source for HTMHT
 *
 *  $Date: 2009/03/12 00:21:12 $
 *  $Revision: 1.1 $
 *  \author K. Hatakeyama, Rockefeller University
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/src/JetAnalyzerBase.h"
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
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

class HTMHTAnalyzer : public JetAnalyzerBase {
 public:

  /// Constructor
  HTMHTAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~HTMHTAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
               const edm::TriggerResults&,
	       const reco::CaloJetCollection& jetcoll);

  int evtCounter;

 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;

  /// number of Jet or MB HLT trigger paths 
  unsigned int nHLTPathsJetMB_;
  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;
  // list of Jet or MB HLT trigger index
  std::vector<unsigned int> HLTPathsJetMBByIndex_;

  // Pt threshold for Jets
  double _ptThreshold;

  //histo binning parameters
  int    phiBin;
  double phiMin;
  double phiMax;

  int    ptBin;
  double ptMin;
  double ptMax;

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
