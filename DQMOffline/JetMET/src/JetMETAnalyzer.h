#ifndef JetMETAnalyzer_H
#define JetMETAnalyzer_H


/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  $Date:$
 *  $Revision:$
 *  \author F. Chlebana - Fermilab
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
  edm::InputTag thePFJetCollectionLabel;
  edm::InputTag theCaloMETCollectionLabel;
  
  bool theJetAnalyzerFlag;
  bool thePFJetAnalyzerFlag;
  bool theCaloMETAnalyzerFlag;

  // the jet analyzer
  JetAnalyzer       * theJetAnalyzer;
  PFJetAnalyzer     * thePFJetAnalyzer;
  CaloMETAnalyzer   * theCaloMETAnalyzer;
  
};
#endif  
