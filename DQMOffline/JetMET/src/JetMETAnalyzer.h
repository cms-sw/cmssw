#ifndef JetMETAnalyzer_H
#define JetMETAnalyzer_H


/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  $Date: 2008/12/08 12:06:46 $
 *  $Revision: 1.6 $
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
#include "DQMOffline/JetMET/src/JetAnalyzer.h"
#include "DQMOffline/JetMET/src/PFJetAnalyzer.h"
#include "DQMOffline/JetMET/src/CaloMETAnalyzer.h"

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
  edm::InputTag theCaloMETCollectionLabel;
  edm::InputTag theCaloMETNoHFCollectionLabel;
  edm::InputTag theTriggerResultsLabel;
  //

  std::string LoJetTrigger;
  std::string HiJetTrigger;
  
  bool theJetAnalyzerFlag;
  bool thePFJetAnalyzerFlag;
  bool theJPTJetAnalyzerFlag;
  bool theCaloMETAnalyzerFlag;

  // the jet analyzer
  JetAnalyzer       * theJetAnalyzer;
  JetAnalyzer       * theSCJetAnalyzer;
  JetAnalyzer       * theICJetAnalyzer;
  JetAnalyzer       * theJPTJetAnalyzer;
  PFJetAnalyzer     * thePFJetAnalyzer;
  CaloMETAnalyzer   * theCaloMETAnalyzer;
  
};
#endif  
