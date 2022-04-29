/*
  JetMETHLTOffline DQM code
  Migrated to use DQMEDAnalyzer by: Jyothsna Rani Komaragiri, Oct 2014
*/

#ifndef JetMETHLTOfflineSource_H
#define JetMETHLTOfflineSource_H

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "RecoJets/JetProducers/interface/JetIDHelper.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <utility>
#include <vector>
#include <string>
#include <sstream>

class PtSorter {
public:
  template <class T>
  bool operator()(const T& a, const T& b) {
    return (a.pt() > b.pt());
  }
};

class JetMETHLTOfflineSource : public DQMEDAnalyzer {
public:
  explicit JetMETHLTOfflineSource(const edm::ParameterSet&);
  ~JetMETHLTOfflineSource() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const& run, edm::EventSetup const& c) override;
  void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override;

  //helper functions
  virtual bool isBarrel(double eta);
  virtual bool isEndCap(double eta);
  virtual bool isForward(double eta);
  virtual bool validPathHLT(std::string path);
  virtual bool isHLTPathAccepted(std::string pathName);
  virtual bool isTriggerObjectFound(std::string objectName);
  virtual double TriggerPosition(std::string trigName);

  virtual void fillMEforMonTriggerSummary(const edm::Event& iEvent, const edm::EventSetup&);
  virtual void fillMEforMonAllTrigger(const edm::Event& iEvent, const edm::EventSetup&);
  virtual void fillMEforEffAllTrigger(const edm::Event& iEvent, const edm::EventSetup&);
  virtual void fillMEforTriggerNTfired();

  const std::string getL1ConditionModuleName(const std::string& pathname);  //ml added

  // ----------member data ---------------------------
  std::vector<std::string> MuonTrigPaths_;
  std::vector<std::string> MBTrigPaths_;
  std::vector<int> prescUsed_;

  std::string dirname_;
  std::string processname_;

  // JetID helper
  reco::helper::JetIDHelper* jetID;

  bool verbose_;
  bool runStandalone_;
  bool plotAll_;
  bool plotEff_;

  bool isSetup_;
  bool nameForEff_;

  double _fEMF;
  double _feta;
  double _fHPD;
  double _n90Hits;
  double _pfMHT;
  double _min_NHEF;
  double _max_NHEF;
  double _min_CHEF;
  double _max_CHEF;
  double _min_NEMF;
  double _max_NEMF;
  double _min_CEMF;
  double _max_CEMF;

  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultsLabel_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsFUToken;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryToken;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryFUToken;
  edm::EDGetTokenT<reco::PFJetCollection> pfJetsToken;
  edm::EDGetTokenT<reco::PFMETCollection> pfMetToken;
  edm::EDGetTokenT<reco::CaloJetCollection> caloJetsToken;
  edm::EDGetTokenT<reco::CaloMETCollection> caloMetToken;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken;

  edm::Handle<reco::CaloJetCollection> calojetColl_;
  edm::Handle<reco::CaloMETCollection> calometColl_;
  edm::Handle<reco::PFJetCollection> pfjetColl_;
  edm::Handle<reco::PFMETCollection> pfmetColl_;

  edm::EDGetTokenT<reco::JetCorrector> CaloJetCorToken_;
  edm::EDGetTokenT<reco::JetCorrector> PFJetCorToken_;

  std::vector<std::string> pathFilter_;
  std::vector<std::string> pathRejectKeyword_;
  std::vector<std::pair<std::string, std::string> > custompathnamepairs_;

  reco::CaloJetCollection calojet;
  reco::PFJetCollection pfjet;
  HLTConfigProvider hltConfig_;
  edm::Handle<edm::TriggerResults> triggerResults_;
  edm::TriggerNames triggerNames_;  // TriggerNames class
  edm::Handle<trigger::TriggerEvent> triggerObj_;

  double CaloJetPx[2];
  double CaloJetPy[2];
  double CaloJetPt[2];
  double CaloJetEta[2];
  double CaloJetPhi[2];
  double CaloJetEMF[2];
  double CaloJetfHPD[2];
  double CaloJetn90[2];

  double PFJetPx[2];
  double PFJetPy[2];
  double PFJetPt[2];
  double PFJetEta[2];
  double PFJetPhi[2];
  double PFJetNHEF[2];
  double PFJetCHEF[2];
  double PFJetNEMF[2];
  double PFJetCEMF[2];

  double pfMHTx_All;
  double pfMHTy_All;

  // ----------------- //
  // helper class to store the data path

  class PathInfo {
    PathInfo()
        : prescaleUsed_(-1),
          denomPathName_("unset"),
          pathName_("unset"),
          l1pathName_("unset"),
          filterName_("unset"),
          DenomfilterName_("unset"),
          processName_("unset"),
          objectType_(-1),
          triggerType_("unset"){};

  public:
    void setHistos(MonitorElement* const N,
                   MonitorElement* const Pt,
                   MonitorElement* const PtBarrel,
                   MonitorElement* const PtEndcap,
                   MonitorElement* const PtForward,
                   MonitorElement* const Eta,
                   MonitorElement* const Phi,
                   MonitorElement* const EtaPhi,
                   //
                   MonitorElement* const N_L1,
                   MonitorElement* const Pt_L1,
                   MonitorElement* const PtBarrel_L1,
                   MonitorElement* const PtEndcap_L1,
                   MonitorElement* const PtForward_L1,
                   MonitorElement* const Eta_L1,
                   MonitorElement* const Phi_L1,
                   MonitorElement* const EtaPhi_L1,
                   MonitorElement* const N_HLT,
                   MonitorElement* const Pt_HLT,
                   MonitorElement* const PtBarrel_HLT,
                   MonitorElement* const PtEndcap_HLT,
                   MonitorElement* const PtForward_HLT,
                   MonitorElement* const Eta_HLT,
                   MonitorElement* const Phi_HLT,
                   MonitorElement* const EtaPhi_HLT,
                   //
                   MonitorElement* const PtResolution_L1HLT,
                   MonitorElement* const EtaResolution_L1HLT,
                   MonitorElement* const PhiResolution_L1HLT,
                   MonitorElement* const PtResolution_HLTRecObj,
                   MonitorElement* const EtaResolution_HLTRecObj,
                   MonitorElement* const PhiResolution_HLTRecObj,
                   //
                   MonitorElement* const PtCorrelation_L1HLT,
                   MonitorElement* const EtaCorrelation_L1HLT,
                   MonitorElement* const PhiCorrelation_L1HLT,
                   MonitorElement* const PtCorrelation_HLTRecObj,
                   MonitorElement* const EtaCorrelation_HLTRecObj,
                   MonitorElement* const PhiCorrelation_HLTRecObj,
                   //
                   MonitorElement* const JetAveragePt,
                   MonitorElement* const JetAverageEta,
                   MonitorElement* const JetPhiDifference,
                   MonitorElement* const HLTAveragePt,
                   MonitorElement* const HLTAverageEta,
                   MonitorElement* const HLTPhiDifference,
                   MonitorElement* const L1AveragePt,
                   MonitorElement* const L1AverageEta,
                   MonitorElement* const L1PhiDifference)

    {
      N_ = N;
      Pt_ = Pt;
      PtBarrel_ = PtBarrel;
      PtEndcap_ = PtEndcap;
      PtForward_ = PtForward;
      Eta_ = Eta;
      Phi_ = Phi;
      EtaPhi_ = EtaPhi;
      N_L1_ = N_L1;
      Pt_L1_ = Pt_L1;
      PtBarrel_L1_ = PtBarrel_L1;
      PtEndcap_L1_ = PtEndcap_L1;
      PtForward_L1_ = PtForward_L1;
      Eta_L1_ = Eta_L1;
      Phi_L1_ = Phi_L1;
      EtaPhi_L1_ = EtaPhi_L1;
      N_HLT_ = N_HLT;
      Pt_HLT_ = Pt_HLT;
      PtBarrel_HLT_ = PtBarrel_HLT;
      PtEndcap_HLT_ = PtEndcap_HLT;
      PtForward_HLT_ = PtForward_HLT;
      Eta_HLT_ = Eta_HLT;
      Phi_HLT_ = Phi_HLT;
      EtaPhi_HLT_ = EtaPhi_HLT;
      //
      PtResolution_L1HLT_ = PtResolution_L1HLT;
      EtaResolution_L1HLT_ = EtaResolution_L1HLT;
      PhiResolution_L1HLT_ = PhiResolution_L1HLT;
      PtResolution_HLTRecObj_ = PtResolution_HLTRecObj;
      EtaResolution_HLTRecObj_ = EtaResolution_HLTRecObj;
      PhiResolution_HLTRecObj_ = PhiResolution_HLTRecObj;
      //
      PtCorrelation_L1HLT_ = PtCorrelation_L1HLT;
      EtaCorrelation_L1HLT_ = EtaCorrelation_L1HLT;
      PhiCorrelation_L1HLT_ = PhiCorrelation_L1HLT;
      PtCorrelation_HLTRecObj_ = PtCorrelation_HLTRecObj;
      EtaCorrelation_HLTRecObj_ = EtaCorrelation_HLTRecObj;
      PhiCorrelation_HLTRecObj_ = PhiCorrelation_HLTRecObj;
      //
      JetAveragePt_ = JetAveragePt;
      JetAverageEta_ = JetAverageEta;
      JetPhiDifference_ = JetPhiDifference;
      HLTAveragePt_ = HLTAveragePt;
      HLTAverageEta_ = HLTAverageEta;
      HLTPhiDifference_ = HLTPhiDifference;
      L1AveragePt_ = L1AveragePt;
      L1AverageEta_ = L1AverageEta;
      L1PhiDifference_ = L1PhiDifference;
    };

    void setDgnsHistos(MonitorElement* const TriggerSummary,
                       MonitorElement* const JetSize,
                       MonitorElement* const JetPt,
                       MonitorElement* const EtavsPt,
                       MonitorElement* const PhivsPt,
                       MonitorElement* const Pt12,
                       MonitorElement* const Eta12,
                       MonitorElement* const Phi12,
                       MonitorElement* const Pt3,
                       MonitorElement* const Pt12Pt3,
                       MonitorElement* const Pt12Phi12) {
      TriggerSummary_ = TriggerSummary;
      JetSize_ = JetSize;
      JetPt_ = JetPt;
      EtavsPt_ = EtavsPt;
      PhivsPt_ = PhivsPt;
      Pt12_ = Pt12;
      Eta12_ = Eta12;
      Phi12_ = Phi12;
      Pt3_ = Pt3;
      Pt12Pt3_ = Pt12Pt3;
      Pt12Phi12_ = Pt12Phi12;
    };

    void setEffHistos(MonitorElement* const NumeratorPt,
                      MonitorElement* const NumeratorPtBarrel,
                      MonitorElement* const NumeratorPtEndcap,
                      MonitorElement* const NumeratorPtForward,
                      MonitorElement* const NumeratorEta,
                      MonitorElement* const NumeratorPhi,
                      MonitorElement* const NumeratorEtaPhi,
                      //
                      MonitorElement* const NumeratorEtaBarrel,
                      MonitorElement* const NumeratorPhiBarrel,
                      MonitorElement* const NumeratorEtaEndcap,
                      MonitorElement* const NumeratorPhiEndcap,
                      MonitorElement* const NumeratorEtaForward,
                      MonitorElement* const NumeratorPhiForward,
                      MonitorElement* const NumeratorEta_LowpTcut,
                      MonitorElement* const NumeratorPhi_LowpTcut,
                      MonitorElement* const NumeratorEtaPhi_LowpTcut,
                      MonitorElement* const NumeratorEta_MedpTcut,
                      MonitorElement* const NumeratorPhi_MedpTcut,
                      MonitorElement* const NumeratorEtaPhi_MedpTcut,
                      MonitorElement* const NumeratorEta_HighpTcut,
                      MonitorElement* const NumeratorPhi_HighpTcut,
                      MonitorElement* const NumeratorEtaPhi_HighpTcut,
                      //
                      MonitorElement* const DenominatorPt,
                      MonitorElement* const DenominatorPtBarrel,
                      MonitorElement* const DenominatorPtEndcap,
                      MonitorElement* const DenominatorPtForward,
                      MonitorElement* const DenominatorEta,
                      MonitorElement* const DenominatorPhi,
                      MonitorElement* const DenominatorEtaPhi,
                      //
                      MonitorElement* const DenominatorEtaBarrel,
                      MonitorElement* const DenominatorPhiBarrel,
                      MonitorElement* const DenominatorEtaEndcap,
                      MonitorElement* const DenominatorPhiEndcap,
                      MonitorElement* const DenominatorEtaForward,
                      MonitorElement* const DenominatorPhiForward,
                      MonitorElement* const DenominatorEta_LowpTcut,
                      MonitorElement* const DenominatorPhi_LowpTcut,
                      MonitorElement* const DenominatorEtaPhi_LowpTcut,
                      MonitorElement* const DenominatorEta_MedpTcut,
                      MonitorElement* const DenominatorPhi_MedpTcut,
                      MonitorElement* const DenominatorEtaPhi_MedpTcut,
                      MonitorElement* const DenominatorEta_HighpTcut,
                      MonitorElement* const DenominatorPhi_HighpTcut,
                      MonitorElement* const DenominatorEtaPhi_HighpTcut,
                      //
                      MonitorElement* const DeltaR,
                      MonitorElement* const DeltaPhi,
                      //
                      MonitorElement* const NumeratorPFPt,
                      MonitorElement* const NumeratorPFMHT,
                      MonitorElement* const NumeratorPFPtBarrel,
                      MonitorElement* const NumeratorPFPtEndcap,
                      MonitorElement* const NumeratorPFPtForward,
                      MonitorElement* const NumeratorPFEta,
                      MonitorElement* const NumeratorPFPhi,
                      MonitorElement* const NumeratorPFEtaPhi,
                      //
                      MonitorElement* const NumeratorPFEtaBarrel,
                      MonitorElement* const NumeratorPFPhiBarrel,
                      MonitorElement* const NumeratorPFEtaEndcap,
                      MonitorElement* const NumeratorPFPhiEndcap,
                      MonitorElement* const NumeratorPFEtaForward,
                      MonitorElement* const NumeratorPFPhiForward,
                      MonitorElement* const NumeratorPFEta_LowpTcut,
                      MonitorElement* const NumeratorPFPhi_LowpTcut,
                      MonitorElement* const NumeratorPFEtaPhi_LowpTcut,
                      MonitorElement* const NumeratorPFEta_MedpTcut,
                      MonitorElement* const NumeratorPFPhi_MedpTcut,
                      MonitorElement* const NumeratorPFEtaPhi_MedpTcut,
                      MonitorElement* const NumeratorPFEta_HighpTcut,
                      MonitorElement* const NumeratorPFPhi_HighpTcut,
                      MonitorElement* const NumeratorPFEtaPhi_HighpTcut,
                      //
                      MonitorElement* const DenominatorPFPt,
                      MonitorElement* const DenominatorPFMHT,
                      MonitorElement* const DenominatorPFPtBarrel,
                      MonitorElement* const DenominatorPFPtEndcap,
                      MonitorElement* const DenominatorPFPtForward,
                      MonitorElement* const DenominatorPFEta,
                      MonitorElement* const DenominatorPFPhi,
                      MonitorElement* const DenominatorPFEtaPhi,
                      //
                      MonitorElement* const DenominatorPFEtaBarrel,
                      MonitorElement* const DenominatorPFPhiBarrel,
                      MonitorElement* const DenominatorPFEtaEndcap,
                      MonitorElement* const DenominatorPFPhiEndcap,
                      MonitorElement* const DenominatorPFEtaForward,
                      MonitorElement* const DenominatorPFPhiForward,
                      MonitorElement* const DenominatorPFEta_LowpTcut,
                      MonitorElement* const DenominatorPFPhi_LowpTcut,
                      MonitorElement* const DenominatorPFEtaPhi_LowpTcut,
                      MonitorElement* const DenominatorPFEta_MedpTcut,
                      MonitorElement* const DenominatorPFPhi_MedpTcut,
                      MonitorElement* const DenominatorPFEtaPhi_MedpTcut,
                      MonitorElement* const DenominatorPFEta_HighpTcut,
                      MonitorElement* const DenominatorPFPhi_HighpTcut,
                      MonitorElement* const DenominatorPFEtaPhi_HighpTcut,
                      //
                      MonitorElement* const PFDeltaR,
                      MonitorElement* const PFDeltaPhi) {
      NumeratorPt_ = NumeratorPt;
      NumeratorPtBarrel_ = NumeratorPtBarrel;
      NumeratorPtEndcap_ = NumeratorPtEndcap;
      NumeratorPtForward_ = NumeratorPtForward;
      NumeratorEta_ = NumeratorEta;
      NumeratorPhi_ = NumeratorPhi;
      NumeratorEtaPhi_ = NumeratorEtaPhi;
      //
      NumeratorEtaBarrel_ = NumeratorEtaBarrel;
      NumeratorPhiBarrel_ = NumeratorPhiBarrel;
      NumeratorEtaEndcap_ = NumeratorEtaEndcap;
      NumeratorPhiEndcap_ = NumeratorPhiEndcap;
      NumeratorEtaForward_ = NumeratorEtaForward;
      NumeratorPhiForward_ = NumeratorPhiForward;
      NumeratorEta_LowpTcut_ = NumeratorEta_LowpTcut;
      NumeratorPhi_LowpTcut_ = NumeratorPhi_LowpTcut;
      NumeratorEtaPhi_LowpTcut_ = NumeratorEtaPhi_LowpTcut;
      NumeratorEta_MedpTcut_ = NumeratorEta_MedpTcut;
      NumeratorPhi_MedpTcut_ = NumeratorPhi_MedpTcut;
      NumeratorEtaPhi_MedpTcut_ = NumeratorEtaPhi_MedpTcut;
      NumeratorEta_HighpTcut_ = NumeratorEta_HighpTcut;
      NumeratorPhi_HighpTcut_ = NumeratorPhi_HighpTcut;
      NumeratorEtaPhi_HighpTcut_ = NumeratorEtaPhi_HighpTcut;
      //
      DenominatorPt_ = DenominatorPt;
      DenominatorPtBarrel_ = DenominatorPtBarrel;
      DenominatorPtEndcap_ = DenominatorPtEndcap;
      DenominatorPtForward_ = DenominatorPtForward;
      DenominatorEta_ = DenominatorEta;
      DenominatorPhi_ = DenominatorPhi;
      DenominatorEtaPhi_ = DenominatorEtaPhi;
      //
      DenominatorEtaBarrel_ = DenominatorEtaBarrel;
      DenominatorPhiBarrel_ = DenominatorPhiBarrel;
      DenominatorEtaEndcap_ = DenominatorEtaEndcap;
      DenominatorPhiEndcap_ = DenominatorPhiEndcap;
      DenominatorEtaForward_ = DenominatorEtaForward;
      DenominatorPhiForward_ = DenominatorPhiForward;
      DenominatorEta_LowpTcut_ = DenominatorEta_LowpTcut;
      DenominatorPhi_LowpTcut_ = DenominatorPhi_LowpTcut;
      DenominatorEtaPhi_LowpTcut_ = DenominatorEtaPhi_LowpTcut;
      DenominatorEta_MedpTcut_ = DenominatorEta_MedpTcut;
      DenominatorPhi_MedpTcut_ = DenominatorPhi_MedpTcut;
      DenominatorEtaPhi_MedpTcut_ = DenominatorEtaPhi_MedpTcut;
      DenominatorEta_HighpTcut_ = DenominatorEta_HighpTcut;
      DenominatorPhi_HighpTcut_ = DenominatorPhi_HighpTcut;
      DenominatorEtaPhi_HighpTcut_ = DenominatorEtaPhi_HighpTcut;
      //
      DeltaR_ = DeltaR;
      DeltaPhi_ = DeltaPhi;
      //
      NumeratorPFPt_ = NumeratorPFPt;
      NumeratorPFMHT_ = NumeratorPFMHT;
      NumeratorPFPtBarrel_ = NumeratorPFPtBarrel;
      NumeratorPFPtEndcap_ = NumeratorPFPtEndcap;
      NumeratorPFPtForward_ = NumeratorPFPtForward;
      NumeratorPFEta_ = NumeratorPFEta;
      NumeratorPFPhi_ = NumeratorPFPhi;
      NumeratorPFEtaPhi_ = NumeratorPFEtaPhi;
      //
      NumeratorPFEtaBarrel_ = NumeratorPFEtaBarrel;
      NumeratorPFPhiBarrel_ = NumeratorPFPhiBarrel;
      NumeratorPFEtaEndcap_ = NumeratorPFEtaEndcap;
      NumeratorPFPhiEndcap_ = NumeratorPFPhiEndcap;
      NumeratorPFEtaForward_ = NumeratorPFEtaForward;
      NumeratorPFPhiForward_ = NumeratorPFPhiForward;
      NumeratorPFEta_LowpTcut_ = NumeratorPFEta_LowpTcut;
      NumeratorPFPhi_LowpTcut_ = NumeratorPFPhi_LowpTcut;
      NumeratorPFEtaPhi_LowpTcut_ = NumeratorPFEtaPhi_LowpTcut;
      NumeratorPFEta_MedpTcut_ = NumeratorPFEta_MedpTcut;
      NumeratorPFPhi_MedpTcut_ = NumeratorPFPhi_MedpTcut;
      NumeratorPFEtaPhi_MedpTcut_ = NumeratorPFEtaPhi_MedpTcut;
      NumeratorPFEta_HighpTcut_ = NumeratorPFEta_HighpTcut;
      NumeratorPFPhi_HighpTcut_ = NumeratorPFPhi_HighpTcut;
      NumeratorPFEtaPhi_HighpTcut_ = NumeratorPFEtaPhi_HighpTcut;
      //
      DenominatorPFPt_ = DenominatorPFPt;
      DenominatorPFMHT_ = DenominatorPFMHT;
      DenominatorPFPtBarrel_ = DenominatorPFPtBarrel;
      DenominatorPFPtEndcap_ = DenominatorPFPtEndcap;
      DenominatorPFPtForward_ = DenominatorPFPtForward;
      DenominatorPFEta_ = DenominatorPFEta;
      DenominatorPFPhi_ = DenominatorPFPhi;
      DenominatorPFEtaPhi_ = DenominatorPFEtaPhi;
      //
      DenominatorPFEtaBarrel_ = DenominatorPFEtaBarrel;
      DenominatorPFPhiBarrel_ = DenominatorPFPhiBarrel;
      DenominatorPFEtaEndcap_ = DenominatorPFEtaEndcap;
      DenominatorPFPhiEndcap_ = DenominatorPFPhiEndcap;
      DenominatorPFEtaForward_ = DenominatorPFEtaForward;
      DenominatorPFPhiForward_ = DenominatorPFPhiForward;
      DenominatorPFEta_LowpTcut_ = DenominatorPFEta_LowpTcut;
      DenominatorPFPhi_LowpTcut_ = DenominatorPFPhi_LowpTcut;
      DenominatorPFEtaPhi_LowpTcut_ = DenominatorPFEtaPhi_LowpTcut;
      DenominatorPFEta_MedpTcut_ = DenominatorPFEta_MedpTcut;
      DenominatorPFPhi_MedpTcut_ = DenominatorPFPhi_MedpTcut;
      DenominatorPFEtaPhi_MedpTcut_ = DenominatorPFEtaPhi_MedpTcut;
      DenominatorPFEta_HighpTcut_ = DenominatorPFEta_HighpTcut;
      DenominatorPFPhi_HighpTcut_ = DenominatorPFPhi_HighpTcut;
      DenominatorPFEtaPhi_HighpTcut_ = DenominatorPFEtaPhi_HighpTcut;
      //
      PFDeltaR_ = PFDeltaR;
      PFDeltaPhi_ = PFDeltaPhi;
    };
    ~PathInfo() = default;
    ;
    PathInfo(int prescaleUsed,
             std::string denomPathName,
             std::string pathName,
             std::string l1pathName,
             std::string filterName,
             std::string DenomfilterName,
             std::string processName,
             size_t type,
             std::string triggerType)
        : prescaleUsed_(prescaleUsed),
          denomPathName_(std::move(denomPathName)),
          pathName_(std::move(pathName)),
          l1pathName_(std::move(l1pathName)),
          filterName_(std::move(filterName)),
          DenomfilterName_(std::move(DenomfilterName)),
          processName_(std::move(processName)),
          objectType_(type),
          triggerType_(std::move(triggerType)) {}

    MonitorElement* getMEhisto_N() { return N_; }
    MonitorElement* getMEhisto_Pt() { return Pt_; }
    MonitorElement* getMEhisto_PtBarrel() { return PtBarrel_; }
    MonitorElement* getMEhisto_PtEndcap() { return PtEndcap_; }
    MonitorElement* getMEhisto_PtForward() { return PtForward_; }
    MonitorElement* getMEhisto_Eta() { return Eta_; }
    MonitorElement* getMEhisto_Phi() { return Phi_; }
    MonitorElement* getMEhisto_EtaPhi() { return EtaPhi_; }

    MonitorElement* getMEhisto_N_L1() { return N_L1_; }
    MonitorElement* getMEhisto_Pt_L1() { return Pt_L1_; }
    MonitorElement* getMEhisto_PtBarrel_L1() { return PtBarrel_L1_; }
    MonitorElement* getMEhisto_PtEndcap_L1() { return PtEndcap_L1_; }
    MonitorElement* getMEhisto_PtForward_L1() { return PtForward_L1_; }
    MonitorElement* getMEhisto_Eta_L1() { return Eta_L1_; }
    MonitorElement* getMEhisto_Phi_L1() { return Phi_L1_; }
    MonitorElement* getMEhisto_EtaPhi_L1() { return EtaPhi_L1_; }

    MonitorElement* getMEhisto_N_HLT() { return N_HLT_; }
    MonitorElement* getMEhisto_Pt_HLT() { return Pt_HLT_; }
    MonitorElement* getMEhisto_PtBarrel_HLT() { return PtBarrel_HLT_; }
    MonitorElement* getMEhisto_PtEndcap_HLT() { return PtEndcap_HLT_; }
    MonitorElement* getMEhisto_PtForward_HLT() { return PtForward_HLT_; }
    MonitorElement* getMEhisto_Eta_HLT() { return Eta_HLT_; }
    MonitorElement* getMEhisto_Phi_HLT() { return Phi_HLT_; }
    MonitorElement* getMEhisto_EtaPhi_HLT() { return EtaPhi_HLT_; }

    MonitorElement* getMEhisto_PtResolution_L1HLT() { return PtResolution_L1HLT_; }
    MonitorElement* getMEhisto_EtaResolution_L1HLT() { return EtaResolution_L1HLT_; }
    MonitorElement* getMEhisto_PhiResolution_L1HLT() { return PhiResolution_L1HLT_; }
    MonitorElement* getMEhisto_PtResolution_HLTRecObj() { return PtResolution_HLTRecObj_; }
    MonitorElement* getMEhisto_EtaResolution_HLTRecObj() { return EtaResolution_HLTRecObj_; }
    MonitorElement* getMEhisto_PhiResolution_HLTRecObj() { return PhiResolution_HLTRecObj_; }

    MonitorElement* getMEhisto_PtCorrelation_L1HLT() { return PtCorrelation_L1HLT_; }
    MonitorElement* getMEhisto_EtaCorrelation_L1HLT() { return EtaCorrelation_L1HLT_; }
    MonitorElement* getMEhisto_PhiCorrelation_L1HLT() { return PhiCorrelation_L1HLT_; }
    MonitorElement* getMEhisto_PtCorrelation_HLTRecObj() { return PtCorrelation_HLTRecObj_; }
    MonitorElement* getMEhisto_EtaCorrelation_HLTRecObj() { return EtaCorrelation_HLTRecObj_; }
    MonitorElement* getMEhisto_PhiCorrelation_HLTRecObj() { return PhiCorrelation_HLTRecObj_; }

    MonitorElement* getMEhisto_AveragePt_RecObj() { return JetAveragePt_; }
    MonitorElement* getMEhisto_AverageEta_RecObj() { return JetAverageEta_; }
    MonitorElement* getMEhisto_DeltaPhi_RecObj() { return JetPhiDifference_; }
    MonitorElement* getMEhisto_AveragePt_HLTObj() { return HLTAveragePt_; }
    MonitorElement* getMEhisto_AverageEta_HLTObj() { return HLTAverageEta_; }
    MonitorElement* getMEhisto_DeltaPhi_HLTObj() { return HLTPhiDifference_; }
    MonitorElement* getMEhisto_AveragePt_L1Obj() { return L1AveragePt_; }
    MonitorElement* getMEhisto_AverageEta_L1Obj() { return L1AverageEta_; }
    MonitorElement* getMEhisto_DeltaPhi_L1Obj() { return L1PhiDifference_; }

    MonitorElement* getMEhisto_NumeratorPt() { return NumeratorPt_; }
    MonitorElement* getMEhisto_NumeratorPtBarrel() { return NumeratorPtBarrel_; }
    MonitorElement* getMEhisto_NumeratorPtEndcap() { return NumeratorPtEndcap_; }
    MonitorElement* getMEhisto_NumeratorPtForward() { return NumeratorPtForward_; }
    MonitorElement* getMEhisto_NumeratorEta() { return NumeratorEta_; }
    MonitorElement* getMEhisto_NumeratorPhi() { return NumeratorPhi_; }
    MonitorElement* getMEhisto_NumeratorEtaPhi() { return NumeratorEtaPhi_; }

    //ml
    MonitorElement* getMEhisto_NVertices() { return NVertices_; }
    MonitorElement* getMEhisto_PVZ() { return PVZ_; }

    MonitorElement* getMEhisto_NumeratorEtaBarrel() { return NumeratorEtaBarrel_; }
    MonitorElement* getMEhisto_NumeratorPhiBarrel() { return NumeratorPhiBarrel_; }
    MonitorElement* getMEhisto_NumeratorEtaEndcap() { return NumeratorEtaEndcap_; }
    MonitorElement* getMEhisto_NumeratorPhiEndcap() { return NumeratorPhiEndcap_; }
    MonitorElement* getMEhisto_NumeratorEtaForward() { return NumeratorEtaForward_; }
    MonitorElement* getMEhisto_NumeratorPhiForward() { return NumeratorPhiForward_; }
    MonitorElement* getMEhisto_NumeratorEta_LowpTcut() { return NumeratorEta_LowpTcut_; }
    MonitorElement* getMEhisto_NumeratorPhi_LowpTcut() { return NumeratorPhi_LowpTcut_; }
    MonitorElement* getMEhisto_NumeratorEtaPhi_LowpTcut() { return NumeratorEtaPhi_LowpTcut_; }
    MonitorElement* getMEhisto_NumeratorEta_MedpTcut() { return NumeratorEta_MedpTcut_; }
    MonitorElement* getMEhisto_NumeratorPhi_MedpTcut() { return NumeratorPhi_MedpTcut_; }
    MonitorElement* getMEhisto_NumeratorEtaPhi_MedpTcut() { return NumeratorEtaPhi_MedpTcut_; }
    MonitorElement* getMEhisto_NumeratorEta_HighpTcut() { return NumeratorEta_HighpTcut_; }
    MonitorElement* getMEhisto_NumeratorPhi_HighpTcut() { return NumeratorPhi_HighpTcut_; }
    MonitorElement* getMEhisto_NumeratorEtaPhi_HighpTcut() { return NumeratorEtaPhi_HighpTcut_; }
    //ml

    MonitorElement* getMEhisto_DenominatorPt() { return DenominatorPt_; }
    MonitorElement* getMEhisto_DenominatorPtBarrel() { return DenominatorPtBarrel_; }
    MonitorElement* getMEhisto_DenominatorPtEndcap() { return DenominatorPtEndcap_; }
    MonitorElement* getMEhisto_DenominatorPtForward() { return DenominatorPtForward_; }
    MonitorElement* getMEhisto_DenominatorEta() { return DenominatorEta_; }
    MonitorElement* getMEhisto_DenominatorPhi() { return DenominatorPhi_; }
    MonitorElement* getMEhisto_DenominatorEtaPhi() { return DenominatorEtaPhi_; }

    //ml
    MonitorElement* getMEhisto_DenominatorEtaBarrel() { return DenominatorEtaBarrel_; }
    MonitorElement* getMEhisto_DenominatorPhiBarrel() { return DenominatorPhiBarrel_; }
    MonitorElement* getMEhisto_DenominatorEtaEndcap() { return DenominatorEtaEndcap_; }
    MonitorElement* getMEhisto_DenominatorPhiEndcap() { return DenominatorPhiEndcap_; }
    MonitorElement* getMEhisto_DenominatorEtaForward() { return DenominatorEtaForward_; }
    MonitorElement* getMEhisto_DenominatorPhiForward() { return DenominatorPhiForward_; }

    MonitorElement* getMEhisto_DenominatorEta_LowpTcut() { return DenominatorEta_LowpTcut_; }
    MonitorElement* getMEhisto_DenominatorPhi_LowpTcut() { return DenominatorPhi_LowpTcut_; }
    MonitorElement* getMEhisto_DenominatorEtaPhi_LowpTcut() { return DenominatorEtaPhi_LowpTcut_; }
    MonitorElement* getMEhisto_DenominatorEta_MedpTcut() { return DenominatorEta_MedpTcut_; }
    MonitorElement* getMEhisto_DenominatorPhi_MedpTcut() { return DenominatorPhi_MedpTcut_; }
    MonitorElement* getMEhisto_DenominatorEtaPhi_MedpTcut() { return DenominatorEtaPhi_MedpTcut_; }
    MonitorElement* getMEhisto_DenominatorEta_HighpTcut() { return DenominatorEta_HighpTcut_; }
    MonitorElement* getMEhisto_DenominatorPhi_HighpTcut() { return DenominatorPhi_HighpTcut_; }
    MonitorElement* getMEhisto_DenominatorEtaPhi_HighpTcut() { return DenominatorEtaPhi_HighpTcut_; }
    //adding PF histos SJ
    MonitorElement* getMEhisto_NumeratorPFPt() { return NumeratorPFPt_; }
    MonitorElement* getMEhisto_NumeratorPFMHT() { return NumeratorPFMHT_; }

    MonitorElement* getMEhisto_NumeratorPFPtBarrel() { return NumeratorPFPtBarrel_; }
    MonitorElement* getMEhisto_NumeratorPFPtEndcap() { return NumeratorPFPtEndcap_; }
    MonitorElement* getMEhisto_NumeratorPFPtForward() { return NumeratorPFPtForward_; }
    MonitorElement* getMEhisto_NumeratorPFEta() { return NumeratorPFEta_; }
    MonitorElement* getMEhisto_NumeratorPFPhi() { return NumeratorPFPhi_; }
    MonitorElement* getMEhisto_NumeratorPFEtaPhi() { return NumeratorPFEtaPhi_; }

    MonitorElement* getMEhisto_NumeratorPFEtaBarrel() { return NumeratorPFEtaBarrel_; }
    MonitorElement* getMEhisto_NumeratorPFPhiBarrel() { return NumeratorPFPhiBarrel_; }
    MonitorElement* getMEhisto_NumeratorPFEtaEndcap() { return NumeratorPFEtaEndcap_; }
    MonitorElement* getMEhisto_NumeratorPFPhiEndcap() { return NumeratorPFPhiEndcap_; }
    MonitorElement* getMEhisto_NumeratorPFEtaForward() { return NumeratorPFEtaForward_; }
    MonitorElement* getMEhisto_NumeratorPFPhiForward() { return NumeratorPFPhiForward_; }
    MonitorElement* getMEhisto_NumeratorPFEta_LowpTcut() { return NumeratorPFEta_LowpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFPhi_LowpTcut() { return NumeratorPFPhi_LowpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFEtaPhi_LowpTcut() { return NumeratorPFEtaPhi_LowpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFEta_MedpTcut() { return NumeratorPFEta_MedpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFPhi_MedpTcut() { return NumeratorPFPhi_MedpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFEtaPhi_MedpTcut() { return NumeratorPFEtaPhi_MedpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFEta_HighpTcut() { return NumeratorPFEta_HighpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFPhi_HighpTcut() { return NumeratorPFPhi_HighpTcut_; }
    MonitorElement* getMEhisto_NumeratorPFEtaPhi_HighpTcut() { return NumeratorPFEtaPhi_HighpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFPt() { return DenominatorPFPt_; }
    MonitorElement* getMEhisto_DenominatorPFMHT() { return DenominatorPFMHT_; }
    MonitorElement* getMEhisto_DenominatorPFPtBarrel() { return DenominatorPFPtBarrel_; }
    MonitorElement* getMEhisto_DenominatorPFPtEndcap() { return DenominatorPFPtEndcap_; }
    MonitorElement* getMEhisto_DenominatorPFPtForward() { return DenominatorPFPtForward_; }
    MonitorElement* getMEhisto_DenominatorPFEta() { return DenominatorPFEta_; }
    MonitorElement* getMEhisto_DenominatorPFPhi() { return DenominatorPFPhi_; }
    MonitorElement* getMEhisto_DenominatorPFEtaPhi() { return DenominatorPFEtaPhi_; }

    MonitorElement* getMEhisto_DenominatorPFEtaBarrel() { return DenominatorPFEtaBarrel_; }
    MonitorElement* getMEhisto_DenominatorPFPhiBarrel() { return DenominatorPFPhiBarrel_; }
    MonitorElement* getMEhisto_DenominatorPFEtaEndcap() { return DenominatorPFEtaEndcap_; }
    MonitorElement* getMEhisto_DenominatorPFPhiEndcap() { return DenominatorPFPhiEndcap_; }
    MonitorElement* getMEhisto_DenominatorPFEtaForward() { return DenominatorPFEtaForward_; }
    MonitorElement* getMEhisto_DenominatorPFPhiForward() { return DenominatorPFPhiForward_; }

    MonitorElement* getMEhisto_DenominatorPFEta_LowpTcut() { return DenominatorPFEta_LowpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFPhi_LowpTcut() { return DenominatorPFPhi_LowpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFEtaPhi_LowpTcut() { return DenominatorPFEtaPhi_LowpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFEta_MedpTcut() { return DenominatorPFEta_MedpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFPhi_MedpTcut() { return DenominatorPFPhi_MedpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFEtaPhi_MedpTcut() { return DenominatorPFEtaPhi_MedpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFEta_HighpTcut() { return DenominatorPFEta_HighpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFPhi_HighpTcut() { return DenominatorPFPhi_HighpTcut_; }
    MonitorElement* getMEhisto_DenominatorPFEtaPhi_HighpTcut() { return DenominatorPFEtaPhi_HighpTcut_; }

    MonitorElement* getMEhisto_DeltaR() { return DeltaR_; }
    MonitorElement* getMEhisto_DeltaPhi() { return DeltaPhi_; }
    MonitorElement* getMEhisto_PFDeltaR() { return PFDeltaR_; }
    MonitorElement* getMEhisto_PFDeltaPhi() { return PFDeltaPhi_; }

    MonitorElement* getMEhisto_TriggerSummary() { return TriggerSummary_; }
    MonitorElement* getMEhisto_JetSize() { return JetSize_; }
    MonitorElement* getMEhisto_JetPt() { return JetPt_; }
    MonitorElement* getMEhisto_EtavsPt() { return EtavsPt_; }
    MonitorElement* getMEhisto_PhivsPt() { return PhivsPt_; }
    MonitorElement* getMEhisto_Pt12() { return Pt12_; }
    MonitorElement* getMEhisto_Eta12() { return Eta12_; }
    MonitorElement* getMEhisto_Phi12() { return Phi12_; }
    MonitorElement* getMEhisto_Pt3() { return Pt3_; }
    MonitorElement* getMEhisto_Pt12Pt3() { return Pt12Pt3_; }
    MonitorElement* getMEhisto_Pt12Phi12() { return Pt12Phi12_; }

    const std::string getLabel() const { return filterName_; }
    const std::string getDenomLabel() const { return DenomfilterName_; }

    void setLabel(std::string labelName) {
      filterName_ = std::move(labelName);
      return;
    }
    void setDenomLabel(std::string labelName) {
      DenomfilterName_ = std::move(labelName);
      return;
    }
    const std::string getPath() const { return pathName_; }
    const std::string getl1Path() const { return l1pathName_; }
    const std::string getDenomPath() const { return denomPathName_; }
    const int getprescaleUsed() const { return prescaleUsed_; }
    const std::string getProcess() const { return processName_; }
    const int getObjectType() const { return objectType_; }
    const std::string getTriggerType() const { return triggerType_; }
    const edm::InputTag getTag() const {
      edm::InputTag tagName(filterName_, "", processName_);
      return tagName;
    }
    const edm::InputTag getDenomTag() const {
      edm::InputTag tagName(DenomfilterName_, "", processName_);
      return tagName;
    }
    bool operator==(const std::string& v) { return v == pathName_; }

  private:
    int prescaleUsed_;
    std::string denomPathName_;
    std::string pathName_;
    std::string l1pathName_;
    std::string filterName_;
    std::string DenomfilterName_;
    std::string processName_;
    int objectType_;
    std::string triggerType_;

    MonitorElement* N_;
    MonitorElement* Pt_;
    MonitorElement* PtBarrel_;
    MonitorElement* PtEndcap_;
    MonitorElement* PtForward_;
    MonitorElement* Eta_;
    MonitorElement* Phi_;
    MonitorElement* EtaPhi_;
    MonitorElement* N_L1_;
    MonitorElement* Pt_L1_;
    MonitorElement* PtBarrel_L1_;
    MonitorElement* PtEndcap_L1_;
    MonitorElement* PtForward_L1_;
    MonitorElement* Eta_L1_;
    MonitorElement* Phi_L1_;
    MonitorElement* EtaPhi_L1_;
    MonitorElement* N_HLT_;
    MonitorElement* Pt_HLT_;
    MonitorElement* PtBarrel_HLT_;
    MonitorElement* PtEndcap_HLT_;
    MonitorElement* PtForward_HLT_;
    MonitorElement* Eta_HLT_;
    MonitorElement* Phi_HLT_;
    MonitorElement* EtaPhi_HLT_;

    MonitorElement* PtResolution_L1HLT_;
    MonitorElement* EtaResolution_L1HLT_;
    MonitorElement* PhiResolution_L1HLT_;
    MonitorElement* PtResolution_HLTRecObj_;
    MonitorElement* EtaResolution_HLTRecObj_;
    MonitorElement* PhiResolution_HLTRecObj_;
    MonitorElement* PtCorrelation_L1HLT_;
    MonitorElement* EtaCorrelation_L1HLT_;
    MonitorElement* PhiCorrelation_L1HLT_;
    MonitorElement* PtCorrelation_HLTRecObj_;
    MonitorElement* EtaCorrelation_HLTRecObj_;
    MonitorElement* PhiCorrelation_HLTRecObj_;

    MonitorElement* JetAveragePt_;
    MonitorElement* JetAverageEta_;
    MonitorElement* JetPhiDifference_;
    MonitorElement* HLTAveragePt_;
    MonitorElement* HLTAverageEta_;
    MonitorElement* HLTPhiDifference_;
    MonitorElement* L1AveragePt_;
    MonitorElement* L1AverageEta_;
    MonitorElement* L1PhiDifference_;

    MonitorElement* NumeratorPt_;
    MonitorElement* NumeratorPtBarrel_;
    MonitorElement* NumeratorPtEndcap_;
    MonitorElement* NumeratorPtForward_;
    MonitorElement* NumeratorEta_;
    MonitorElement* NumeratorPhi_;
    MonitorElement* NumeratorEtaPhi_;

    //ml
    MonitorElement* PVZ_;
    MonitorElement* NVertices_;

    MonitorElement* NumeratorEtaBarrel_;
    MonitorElement* NumeratorPhiBarrel_;
    MonitorElement* NumeratorEtaEndcap_;
    MonitorElement* NumeratorPhiEndcap_;
    MonitorElement* NumeratorEtaForward_;
    MonitorElement* NumeratorPhiForward_;

    MonitorElement* NumeratorEta_LowpTcut_;
    MonitorElement* NumeratorPhi_LowpTcut_;
    MonitorElement* NumeratorEtaPhi_LowpTcut_;
    MonitorElement* NumeratorEta_MedpTcut_;
    MonitorElement* NumeratorPhi_MedpTcut_;
    MonitorElement* NumeratorEtaPhi_MedpTcut_;
    MonitorElement* NumeratorEta_HighpTcut_;
    MonitorElement* NumeratorPhi_HighpTcut_;
    MonitorElement* NumeratorEtaPhi_HighpTcut_;
    //ml

    MonitorElement* DenominatorPt_;
    MonitorElement* DenominatorPtBarrel_;
    MonitorElement* DenominatorPtEndcap_;
    MonitorElement* DenominatorPtForward_;
    MonitorElement* DenominatorEta_;
    MonitorElement* DenominatorPhi_;
    MonitorElement* DenominatorEtaPhi_;
    //ml
    MonitorElement* DenominatorEtaBarrel_;
    MonitorElement* DenominatorPhiBarrel_;
    MonitorElement* DenominatorEtaEndcap_;
    MonitorElement* DenominatorPhiEndcap_;
    MonitorElement* DenominatorEtaForward_;
    MonitorElement* DenominatorPhiForward_;

    MonitorElement* DenominatorEta_LowpTcut_;
    MonitorElement* DenominatorPhi_LowpTcut_;
    MonitorElement* DenominatorEtaPhi_LowpTcut_;
    MonitorElement* DenominatorEta_MedpTcut_;
    MonitorElement* DenominatorPhi_MedpTcut_;
    MonitorElement* DenominatorEtaPhi_MedpTcut_;
    MonitorElement* DenominatorEta_HighpTcut_;
    MonitorElement* DenominatorPhi_HighpTcut_;
    MonitorElement* DenominatorEtaPhi_HighpTcut_;

    MonitorElement* DeltaR_;
    MonitorElement* DeltaPhi_;

    //adding PF histos SJ:
    MonitorElement* NumeratorPFPt_;
    MonitorElement* NumeratorPFMHT_;
    MonitorElement* NumeratorPFPtBarrel_;
    MonitorElement* NumeratorPFPtEndcap_;
    MonitorElement* NumeratorPFPtForward_;
    MonitorElement* NumeratorPFEta_;
    MonitorElement* NumeratorPFPhi_;
    MonitorElement* NumeratorPFEtaPhi_;
    MonitorElement* NumeratorPFEtaBarrel_;
    MonitorElement* NumeratorPFPhiBarrel_;
    MonitorElement* NumeratorPFEtaEndcap_;
    MonitorElement* NumeratorPFPhiEndcap_;
    MonitorElement* NumeratorPFEtaForward_;
    MonitorElement* NumeratorPFPhiForward_;

    MonitorElement* NumeratorPFEta_LowpTcut_;
    MonitorElement* NumeratorPFPhi_LowpTcut_;
    MonitorElement* NumeratorPFEtaPhi_LowpTcut_;
    MonitorElement* NumeratorPFEta_MedpTcut_;
    MonitorElement* NumeratorPFPhi_MedpTcut_;
    MonitorElement* NumeratorPFEtaPhi_MedpTcut_;
    MonitorElement* NumeratorPFEta_HighpTcut_;
    MonitorElement* NumeratorPFPhi_HighpTcut_;
    MonitorElement* NumeratorPFEtaPhi_HighpTcut_;

    MonitorElement* DenominatorPFPt_;
    MonitorElement* DenominatorPFMHT_;
    MonitorElement* DenominatorPFPtBarrel_;
    MonitorElement* DenominatorPFPtEndcap_;
    MonitorElement* DenominatorPFPtForward_;
    MonitorElement* DenominatorPFEta_;
    MonitorElement* DenominatorPFPhi_;
    MonitorElement* DenominatorPFEtaPhi_;

    MonitorElement* DenominatorPFEtaBarrel_;
    MonitorElement* DenominatorPFPhiBarrel_;
    MonitorElement* DenominatorPFEtaEndcap_;
    MonitorElement* DenominatorPFPhiEndcap_;
    MonitorElement* DenominatorPFEtaForward_;
    MonitorElement* DenominatorPFPhiForward_;

    MonitorElement* DenominatorPFEta_LowpTcut_;
    MonitorElement* DenominatorPFPhi_LowpTcut_;
    MonitorElement* DenominatorPFEtaPhi_LowpTcut_;
    MonitorElement* DenominatorPFEta_MedpTcut_;
    MonitorElement* DenominatorPFPhi_MedpTcut_;
    MonitorElement* DenominatorPFEtaPhi_MedpTcut_;
    MonitorElement* DenominatorPFEta_HighpTcut_;
    MonitorElement* DenominatorPFPhi_HighpTcut_;
    MonitorElement* DenominatorPFEtaPhi_HighpTcut_;
    MonitorElement* PFDeltaR_;
    MonitorElement* PFDeltaPhi_;

    MonitorElement* TriggerSummary_;
    MonitorElement* JetSize_;
    MonitorElement* JetPt_;
    MonitorElement* EtavsPt_;
    MonitorElement* PhivsPt_;
    MonitorElement* Pt12_;
    MonitorElement* Eta12_;
    MonitorElement* Phi12_;
    MonitorElement* Pt3_;
    MonitorElement* Pt12Pt3_;
    MonitorElement* Pt12Phi12_;
  };

  // simple collection
  class PathInfoCollection : public std::vector<PathInfo> {
  public:
    PathInfoCollection() : std::vector<PathInfo>(){};
    std::vector<PathInfo>::iterator find(const std::string& pathName) { return std::find(begin(), end(), pathName); }
  };
  PathInfoCollection hltPathsAllTriggerSummary_;
  PathInfoCollection hltPathsAll_;
  PathInfoCollection hltPathsEff_;

  MonitorElement* rate_All;
  MonitorElement* rate_AllWrtMu;
  MonitorElement* rate_AllWrtMB;

  MonitorElement* correlation_All;
  MonitorElement* correlation_AllWrtMu;
  MonitorElement* correlation_AllWrtMB;
  MonitorElement* PVZ;
  MonitorElement* NVertices;
};
#endif
