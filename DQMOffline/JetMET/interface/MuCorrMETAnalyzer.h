#ifndef MuCorrMETAnalyzer_H
#define MuCorrMETAnalyzer_H


/** \class MuCorrMETAnalyzer
 *
 *  DQM monitoring source for CaloMET
 *
 *  $Date: 2012/05/20 13:11:46 $
 *  $Revision: 1.9 $
 *  \author A.Apresyan - Caltech
 */


#include <memory>
#include <fstream>
#include "TMath.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/MuCorrMETAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"

class MuCorrMETAnalyzer : public MuCorrMETAnalyzerBase {
 public:

  /// Constructor
  MuCorrMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuCorrMETAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore * dbe);

  /// Finish up a job
  void endJob();

  // Book MonitorElements
  void bookMESet(std::string);
  void bookMonitorElement(std::string, bool);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, 
               const edm::TriggerResults&);

  /// Initialize run-based parameters
  void beginRun(const edm::Run&,  const edm::EventSetup&);

  /// Finish up a run
  void endRun(const edm::Run& iRun, const edm::EventSetup& iSetup, DQMStore *dbe);

  // Fill MonitorElements
  void fillMESet(const edm::Event&, std::string, const reco::CaloMET&);
  void fillMonitorElement(const edm::Event&, std::string, std::string, const reco::CaloMET&, bool);
  void makeRatePlot(std::string, double);

  bool selectHighPtJetEvent(const edm::Event&);
  bool selectLowPtJetEvent(const edm::Event&);
  bool selectWElectronEvent(const edm::Event&);
  bool selectWMuonEvent(const edm::Event&);

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

  edm::InputTag theMuCorrMETCollectionLabel;
  edm::InputTag HcalNoiseRBXCollectionTag;
  edm::InputTag HBHENoiseFilterResultTag;
  edm::InputTag theJetCollectionLabel;
  edm::InputTag thePfJetCollectionLabel;

  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;

  std::string _hlt_HighPtJet;
  std::string _hlt_LowPtJet;
  std::string _hlt_HighMET;
  //  std::string _hlt_LowMET;
  std::string _hlt_Ele;
  std::string _hlt_Muon;
  
  int _trig_JetMB;
  int _trig_HighPtJet;
  int _trig_LowPtJet;
  int _trig_HighMET;
  //  int _trig_LowMET;
  int _trig_Ele;
  int _trig_Muon;


  double _highPtMuCorrJetThreshold;
  double _lowPtMuCorrJetThreshold;
  double _highMuCorrMETThreshold;
  //  double _lowMuCorrMETThreshold;

  // Et threshold for MET plots
  double _etThreshold;

  // HF calibration factor (in 31X applied by MuonProducer)
  double hfCalibFactor_;  //

  // JetID helper
  reco::helper::JetIDHelper *jetID;

  //
  bool _allhist;
  bool _allSelection;

  //
  std::vector<std::string> _FolderNames;

  //
  DQMStore *_dbe;

  //the histos
  MonitorElement* metME;

  MonitorElement* meTriggerName_HighPtJet;
  MonitorElement* meTriggerName_LowPtJet;
  MonitorElement* meTriggerName_HighMET;
  //  MonitorElement* meTriggerName_LowMET;
  MonitorElement* meTriggerName_Ele;
  MonitorElement* meTriggerName_Muon;

  MonitorElement* meMuCorrMEx;
  MonitorElement* meMuCorrMEy;
  MonitorElement* meMuCorrMETSig;
  MonitorElement* meMuCorrMET;
  MonitorElement* meMuCorrMETPhi;
  MonitorElement* meMuCorrSumET;
  MonitorElement* meMuCorrMExLS;
  MonitorElement* meMuCorrMEyLS;

  MonitorElement* meMuCorrMETIonFeedbck;
  MonitorElement* meMuCorrMETHPDNoise;
  MonitorElement* meMuCorrMETRBXNoise;

  MonitorElement* meMuCorrMETRate;
};
#endif
