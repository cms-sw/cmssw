#ifndef PFMETAnalyzer_H
#define PFMETAnalyzer_H


/** \class PFMETAnalyzer
 *
 *  DQM monitoring source for CaloMET
 *
 *  $Date: 2009/06/30 13:48:23 $
 *  $Revision: 1.1 $
 *  \author K. Hatakeyama - Rockefeller University
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/PFMETAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
//
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

class PFMETAnalyzer : public PFMETAnalyzerBase {
 public:

  /// Constructor
  PFMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~PFMETAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
               const edm::TriggerResults&);

  void setSource(std::string source) {
    _source = source;
  }

  int evtCounter;

 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  int _verbose;

  std::string metname;
  std::string _source;

  edm::InputTag thePfMETCollectionLabel;

  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;

  int _trig_JetMB;

  // Et threshold for MET plots
  double _etThreshold;

  //the histos
  MonitorElement* metME;

  MonitorElement* hNevents;
  MonitorElement* hPfMEx;
  MonitorElement* hPfMEy;
  MonitorElement* hPfEz;
  MonitorElement* hPfMETSig;
  MonitorElement* hPfMET;
  MonitorElement* hPfMETPhi;
  MonitorElement* hPfSumET;

  MonitorElement* hPfNeutralEMFraction;
  MonitorElement* hPfNeutralHadFraction;
  MonitorElement* hPfChargedEMFraction;
  MonitorElement* hPfChargedHadFraction;
  MonitorElement* hPfMuonFraction;

};
#endif
