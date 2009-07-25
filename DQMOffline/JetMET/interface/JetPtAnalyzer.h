#ifndef JetPtAnalyzer_H
#define JetPtAnalyzer_H


/** \class JetPtAnalyzer
 *
 *  DQM monitoring source for Calo Jets
 *
 
 *  \author J.Weng ETH Zuerich
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/JetAnalyzerBase.h"
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


#include <string>
using namespace std;
using namespace edm;

class JetPtAnalyzer : public JetAnalyzerBase {
 public:

  /// Constructor
  //  JetPtAnalyzer(const edm::ParameterSet&, JetServiceProxy *theService);
  JetPtAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~JetPtAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

  /// Get the analysis
    //  void analyze(const edm::Event&, const edm::EventSetup&, const edm::TriggerResults&,
    //	       const reco::CaloJet& caloJet);
  void analyze(const edm::Event&, const edm::EventSetup&, 
	       const reco::CaloJet& caloJet);

  void setSource(std::string source) {
    _source = source;
  }

   void setNJets(int njets) {
    _NJets = njets;
  }
  int getNJets() {
    return  _NJets;
  }


 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string jetname;
  std::string _source;
  // Calo Jet Label
  edm::InputTag theCaloJetCollectionLabel;

 
  

  //histo binning parameters
  int    etaBin;
  double etaMin;
  double etaMax;

  int    phiBin;
  double phiMin;
  double phiMax;

  int    ptBin;
  double ptMin;
  double ptMax;

  int   _NJets;

  //the histos
  MonitorElement* jetME;
  MonitorElement* mEta;
  MonitorElement* mPhi;
 
  MonitorElement* mConstituents;
  MonitorElement* mHFrac;
  MonitorElement* mEFrac;

  MonitorElement* mNJets;

  // CaloJet specific
  MonitorElement* mMaxEInEmTowers;
  MonitorElement* mMaxEInHadTowers;
  MonitorElement* mHadEnergyInHO;
  MonitorElement* mHadEnergyInHB;
  MonitorElement* mHadEnergyInHF;
  MonitorElement* mHadEnergyInHE;
  MonitorElement* mEmEnergyInEB;
  MonitorElement* mEmEnergyInEE;
  MonitorElement* mEmEnergyInHF;
  MonitorElement* mN90;


};
#endif
