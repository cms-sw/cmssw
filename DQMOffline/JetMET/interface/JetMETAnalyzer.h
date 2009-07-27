#ifndef JetMETAnalyzer_H
#define JetMETAnalyzer_H


/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  $Date: 2009/07/27 07:10:02 $
 *  $Revision: 1.3 $
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMOffline/JetMET/interface/JetAnalyzer.h"
#include "DQMOffline/JetMET/interface/JetPtAnalyzer.h"
#include "DQMOffline/JetMET/interface/PFJetAnalyzer.h"
#include "DQMOffline/JetMET/interface/CaloMETAnalyzer.h"
#include "DQMOffline/JetMET/interface/METAnalyzer.h"
#include "DQMOffline/JetMET/interface/PFMETAnalyzer.h"
#include "DQMOffline/JetMET/interface/HTMHTAnalyzer.h"

class JetMETAnalyzer : public edm::EDAnalyzer {
 public:

  /// Constructor
  JetMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~JetMETAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

  /// Save the histos
  void endJob(void);

  /// Initialize run-based parameters
  void beginRun(const edm::Run&,  const edm::EventSetup&);

  /// Finish up a run
  void endRun(const edm::Run&,  const edm::EventSetup&);

 private:
  // ----------member data ---------------------------
  
  DQMStore* dbe;
  edm::ParameterSet parameters;
  std::string metname;

  edm::InputTag theCaloJetCollectionLabel;
  edm::InputTag theSCJetCollectionLabel;
  edm::InputTag theICJetCollectionLabel;
  edm::InputTag thePFJetCollectionLabel;
  edm::InputTag theJPTJetCollectionLabel;
/*   edm::InputTag theCaloMETCollectionLabel; */
/*   edm::InputTag theCaloMETNoHFCollectionLabel; */
/*   edm::InputTag theCaloMETHOCollectionLabel; */
/*   edm::InputTag theCaloMETNoHFHOCollectionLabel; */
  edm::InputTag theTcMETCollectionLabel;
  edm::InputTag thePfMETCollectionLabel;
  edm::InputTag theJetCollectionForHTMHTLabel;
  edm::InputTag theTriggerResultsLabel;
  //

  HLTConfigProvider hltConfig_;
  std::string processname_;

  MonitorElement* hltpathME;
  MonitorElement* lumisecME;

  std::string LoJetTrigger;
  std::string HiJetTrigger;
  
  bool theJetAnalyzerFlag;
  bool theJetPtAnalyzerFlag;
  bool theJetCleaningFlag;
  bool thePFJetAnalyzerFlag;
  bool theJPTJetAnalyzerFlag;
  bool theCaloMETAnalyzerFlag;
  bool theTcMETAnalyzerFlag;
  bool thePfMETAnalyzerFlag;
  bool theHTMHTAnalyzerFlag;

  // the jet analyzer
  JetAnalyzer       * theJetAnalyzer;
  JetAnalyzer       * theSCJetAnalyzer;
  JetAnalyzer       * theICJetAnalyzer; 
  JetAnalyzer       * theCleanedSCJetAnalyzer;
  JetAnalyzer       * theCleanedICJetAnalyzer;
  JetAnalyzer       * theJPTJetAnalyzer;
  PFJetAnalyzer     * thePFJetAnalyzer;
  JetPtAnalyzer     * thePtSCJetAnalyzer;
  JetPtAnalyzer     * thePtICJetAnalyzer;
  CaloMETAnalyzer   * theCaloMETAnalyzer;
  CaloMETAnalyzer   * theCaloMETNoHFAnalyzer;
  CaloMETAnalyzer   * theCaloMETHOAnalyzer;
  CaloMETAnalyzer   * theCaloMETNoHFHOAnalyzer;
  METAnalyzer       * theTcMETAnalyzer;
  PFMETAnalyzer     * thePfMETAnalyzer;
  HTMHTAnalyzer     * theHTMHTAnalyzer;
  
};
#endif  
