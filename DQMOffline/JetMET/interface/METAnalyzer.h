#ifndef METAnalyzer_H
#define METAnalyzer_H


/** \class METAnalyzer
 *
 *  DQM monitoring source for CaloMET
 *
 *  $Date: 2009/06/30 13:48:15 $
 *  $Revision: 1.1 $
 *  \author K. Hatakeyama - Rockefeller University
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/METAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/MET.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

class METAnalyzer : public METAnalyzerBase {
 public:

  /// Constructor
  METAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~METAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

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

  edm::InputTag theTcMETCollectionLabel;

  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;

  int _trig_JetMB;

  // Et threshold for MET plots
  double _etThreshold;

  //the histos
  MonitorElement* jetME;

  MonitorElement* hNevents;
  MonitorElement* hMEx;
  MonitorElement* hMEy;
  MonitorElement* hEz;
  MonitorElement* hMETSig;
  MonitorElement* hMET;
  MonitorElement* hMETPhi;
  MonitorElement* hSumET;

};
#endif
