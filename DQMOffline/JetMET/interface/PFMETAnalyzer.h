#ifndef PFMETAnalyzer_H
#define PFMETAnalyzer_H


/** \class PFMETAnalyzer
 *
 *  DQM monitoring source for CaloMET
 *
 *  $Date: 2009/10/08 10:08:28 $
 *  $Revision: 1.2 $
 *  \author K. Hatakeyama - Rockefeller University
 *          A.Apresyan - Caltech 
 */


#include <memory>
#include <fstream>
#include "TMath.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/PFMETAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoMET/METAlgorithms/interface/HcalNoiseRBXArray.h"
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"

#include "RecoJets/JetAlgorithms/interface/JetIDHelper.h"

class PFMETAnalyzer : public PFMETAnalyzerBase {
 public:

  /// Constructor
  PFMETAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~PFMETAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore *dbe);

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
  void fillMESet(const edm::Event&, std::string, const reco::PFMET&);
  void fillMonitorElement(const edm::Event&, std::string, std::string, const reco::PFMET&, bool);
  void makeRatePlot(std::string, double);

  void validateMET(const reco::PFMET&, edm::Handle<edm::View<PFCandidate> >);

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

  edm::InputTag thePfMETCollectionLabel;
  edm::InputTag HcalNoiseRBXCollectionTag;
  edm::InputTag HcalNoiseSummaryTag;
  edm::InputTag theJetCollectionLabel;
  edm::InputTag thePfJetCollectionLabel;
  edm::InputTag PFCandidatesTag;

  // list of Jet or MB HLT triggers
  std::vector<std::string > HLTPathsJetMBByName_;

  std::string _hlt_HighPtJet;
  std::string _hlt_LowPtJet;
  std::string _hlt_HighMET;
  std::string _hlt_LowMET;
  std::string _hlt_Ele;
  std::string _hlt_Muon;
  
  int _trig_JetMB;
  int _trig_HighPtJet;
  int _trig_LowPtJet;
  int _trig_HighMET;
  int _trig_LowMET;
  int _trig_Ele;
  int _trig_Muon;


  double _highPtPFJetThreshold;
  double _lowPtPFJetThreshold;
  double _highPFMETThreshold;
  double _lowPFMETThreshold;

  // Et threshold for MET plots
  double _etThreshold;

  // HF calibration factor (in 31X applied by PFProducer)
  double hfCalibFactor_;  //

  // JetID helper
  reco::helper::JetIDHelper jetID;

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
  MonitorElement* meTriggerName_LowMET;
  MonitorElement* meTriggerName_Ele;
  MonitorElement* meTriggerName_Muon;

  MonitorElement* mePfNeutralEMFraction;
  MonitorElement* mePfNeutralHadFraction;
  MonitorElement* mePfChargedEMFraction;
  MonitorElement* mePfChargedHadFraction;
  MonitorElement* mePfMuonFraction;

  MonitorElement* meNevents;
  MonitorElement* mePfMEx;
  MonitorElement* mePfMEy;
  MonitorElement* mePfEz;
  MonitorElement* mePfMETSig;
  MonitorElement* mePfMET;
  MonitorElement* mePfMETPhi;
  MonitorElement* mePfSumET;
  MonitorElement* mePfMExLS;
  MonitorElement* mePfMEyLS;

  MonitorElement* mePfMETIonFeedbck;
  MonitorElement* mePfMETHPDNoise;
  MonitorElement* mePfMETRBXNoise;

  MonitorElement* mePfMETRate;


};
#endif
