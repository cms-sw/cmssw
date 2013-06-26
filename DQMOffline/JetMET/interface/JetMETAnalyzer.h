#ifndef JetMETAnalyzer_H
#define JetMETAnalyzer_H


/** \class JetMETAnalyzer
 *
 *  DQM jetMET analysis monitoring
 *
 *  $Date: 2012/05/20 13:11:46 $
 *  $Revision: 1.32 $
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
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
//
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
//
#include "DQMOffline/JetMET/interface/JetAnalyzer.h"
#include "DQMOffline/JetMET/interface/JetPtAnalyzer.h"
#include "DQMOffline/JetMET/interface/PFJetAnalyzer.h"
#include "DQMOffline/JetMET/interface/JPTJetAnalyzer.h"
#include "DQMOffline/JetMET/interface/CaloMETAnalyzer.h"
#include "DQMOffline/JetMET/interface/METAnalyzer.h"
#include "DQMOffline/JetMET/interface/PFMETAnalyzer.h"
#include "DQMOffline/JetMET/interface/HTMHTAnalyzer.h"

#include "DQMOffline/JetMET/interface/JetMETDQMDCSFilter.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Scalers/interface/DcsStatus.h" 

class JetMETAnalyzer : public edm::EDAnalyzer {
 public:

  /// Constructor
  JetMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~JetMETAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(void);

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
  edm::InputTag theAKJetCollectionLabel;
  edm::InputTag theSCJetCollectionLabel;
  edm::InputTag theICJetCollectionLabel;
  edm::InputTag thePFJetCollectionLabel;
  edm::InputTag theJPTJetCollectionLabel;
  edm::InputTag theTriggerResultsLabel;


  //Cleaning parameters
  edm::ParameterSet theCleaningParameters;
  edm::InputTag _theVertexLabel;
  edm::InputTag _theGTLabel;
  std::string _hlt_PhysDec;

  bool _hlt_initialized;

  std::vector<unsigned > _techTrigsAND;
  std::vector<unsigned > _techTrigsOR;
  std::vector<unsigned > _techTrigsNOT;

  bool _doPVCheck;
  bool _doHLTPhysicsOn;

  bool _tightBHFiltering;

  int _nvtx_min;
  int _vtxndof_min;
  int _nvtxtrks_min;
  double _vtxchi2_max;
  double _vtxz_max;
  //

  int _LSBegin;
  int _LSEnd;

  HLTConfigProvider hltConfig_;
  std::string processname_;

  //MonitorElement* hltpathME;
  MonitorElement* lumisecME;
  MonitorElement* cleanupME;
  MonitorElement* verticesME;

  GenericTriggerEventFlag * _HighPtJetEventFlag;
  GenericTriggerEventFlag * _LowPtJetEventFlag;

  std::vector<std::string> highPtJetExpr_;
  std::vector<std::string> lowPtJetExpr_;

  bool theJetAnalyzerFlag;  
  bool theIConeJetAnalyzerFlag;
  bool theSConeJetAnalyzerFlag;
  bool theJetCleaningFlag;

  bool theJetPtAnalyzerFlag;
  bool theJetPtCleaningFlag;

  bool thePFJetAnalyzerFlag;
  bool thePFJetCleaningFlag;

  bool theDiJetSelectionFlag;

  bool theJPTJetAnalyzerFlag;
  bool theJPTJetCleaningFlag;

  bool theCaloMETAnalyzerFlag;

  bool theTcMETAnalyzerFlag;

  bool theMuCorrMETAnalyzerFlag;

  bool thePfMETAnalyzerFlag;

  bool theHTMHTAnalyzerFlag;

  // the jet analyzer
  JetAnalyzer       * theJetAnalyzer;
  JetAnalyzer       * theAKJetAnalyzer;   
  JetAnalyzer       * theSCJetAnalyzer;   
  JetAnalyzer       * theICJetAnalyzer;  
  JetAnalyzer       * theCleanedAKJetAnalyzer;    
  JetAnalyzer       * theCleanedSCJetAnalyzer;
  JetAnalyzer       * theCleanedICJetAnalyzer;
  JetAnalyzer       * theDiJetAnalyzer;  

  JPTJetAnalyzer    * theJPTJetAnalyzer;   
  JPTJetAnalyzer    * theCleanedJPTJetAnalyzer;

  PFJetAnalyzer     * thePFJetAnalyzer;     
  PFJetAnalyzer     * theCleanedPFJetAnalyzer; 
  PFJetAnalyzer     * thePFDiJetAnalyzer;

  JetPtAnalyzer     * thePtAKJetAnalyzer;
  JetPtAnalyzer     * thePtSCJetAnalyzer;
  JetPtAnalyzer     * thePtICJetAnalyzer;
  JetPtAnalyzer     * theCleanedPtAKJetAnalyzer;
  JetPtAnalyzer     * theCleanedPtSCJetAnalyzer;
  JetPtAnalyzer     * theCleanedPtICJetAnalyzer;

  CaloMETAnalyzer   * theCaloMETAnalyzer;
  //CaloMETAnalyzer   * theCaloMETNoHFAnalyzer;
  //CaloMETAnalyzer   * theCaloMETHOAnalyzer;
  //CaloMETAnalyzer   * theCaloMETNoHFHOAnalyzer;
  CaloMETAnalyzer   * theMuCorrMETAnalyzer;

  METAnalyzer       * theTcMETAnalyzer;

  PFMETAnalyzer     * thePfMETAnalyzer;

  HTMHTAnalyzer     * theHTMHTAnalyzer;

  JetMETDQMDCSFilter * DCSFilterCalo;
  JetMETDQMDCSFilter * DCSFilterPF;
  JetMETDQMDCSFilter * DCSFilterJPT;
  JetMETDQMDCSFilter * DCSFilterAll;

};
#endif  
