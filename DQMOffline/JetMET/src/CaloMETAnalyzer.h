#ifndef CaloMETAnalyzer_H
#define CaloMETAnalyzer_H


/** \class CaloMETAnalyzer
 *
 *  DQM monitoring source for CaloMET
 *
 *  $Date: 2008/09/19 15:07:48 $
 *  $Revision: 1.4 $
 *  \author F. Chlebana - Fermilab
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/src/CaloMETAnalyzerBase.h"
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

class CaloMETAnalyzer : public CaloMETAnalyzerBase {
 public:

  /// Constructor
  CaloMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~CaloMETAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
               const edm::TriggerResults&,
	       const reco::CaloMET& caloMET, const reco::CaloMET& caloMETNoHF);

  int evtCounter;

 private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // CaloMET Label
  //edm::InputTag theCaloMETCollectionLabel;
  //edm::InputTag theCaloMETNoHFCollectionLabel;

  /// number of Jet or MB HLT trigger paths 
  unsigned int nHLTPathsJetMB_;
  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;
  // list of Jet or MB HLT trigger index
  std::vector<unsigned int> HLTPathsJetMBByIndex_;

  // Et threshold for MET plots
  double _etThreshold;

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

  //the histos
  MonitorElement* jetME;

  MonitorElement* hNevents;
  MonitorElement* hCaloMEx;
  MonitorElement* hCaloMEy;
  MonitorElement* hCaloEz;
  MonitorElement* hCaloMETSig;
  MonitorElement* hCaloMET;
  MonitorElement* hCaloMETPhi;
  MonitorElement* hCaloSumET;
  MonitorElement* hCaloMExLS;
  MonitorElement* hCaloMEyLS;

  MonitorElement* hCaloMaxEtInEmTowers;
  MonitorElement* hCaloMaxEtInHadTowers;
  MonitorElement* hCaloEtFractionHadronic;
  MonitorElement* hCaloEmEtFraction;

  MonitorElement* hCaloHadEtInHB;
  MonitorElement* hCaloHadEtInHO;
  MonitorElement* hCaloHadEtInHE;
  MonitorElement* hCaloHadEtInHF;
  MonitorElement* hCaloHadEtInEB;
  MonitorElement* hCaloHadEtInEE;
  MonitorElement* hCaloEmEtInHF;
  MonitorElement* hCaloEmEtInEE;
  MonitorElement* hCaloEmEtInEB;

  MonitorElement* hCaloMExNoHF;
  MonitorElement* hCaloMEyNoHF;
  MonitorElement* hCaloEzNoHF;
  MonitorElement* hCaloMETSigNoHF;
  MonitorElement* hCaloMETNoHF;
  MonitorElement* hCaloMETPhiNoHF;
  MonitorElement* hCaloSumETNoHF;
  MonitorElement* hCaloMExNoHFLS;
  MonitorElement* hCaloMEyNoHFLS;

};
#endif
