#ifndef MuonRecoOneHLT_H
#define MuonRecoOneHLT_H


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/Muon/src/MuonAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

class MuonRecoOneHLT : public MuonAnalyzerBase {
 public:

  /// Constructor
  MuonRecoOneHLT(const edm::ParameterSet&, MuonServiceProxy *theService);
  
  /// Destructor
  virtual ~MuonRecoOneHLT();

  /// Inizialize parameters for histo binning
  void beginJob(DQMStore *dbe);
  void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  
  /// Get the analysis
  //  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Muon&, const edm::TriggerResults&);
  void analyze(const edm::Event&, const edm::EventSetup&, const edm::TriggerResults&);

 private:
  // ----------member data ---------------------------
  edm::ParameterSet parameters;
  // Switch for verbosity
  std::string metname;
  // STA Label
  edm::InputTag theMuonCollectionLabel;
  edm::InputTag theSTACollectionLabel;

 //Vertex requirements
  edm::InputTag  vertexTag;
  edm::InputTag  bsTag;
  
  std::vector<std::string> singlemuonExpr_;
  std::vector<std::string> doublemuonExpr_;
  GenericTriggerEventFlag *_SingleMuonEventFlag;
  GenericTriggerEventFlag *_DoubleMuonEventFlag;
  
  //histo binning parameters
  int ptBin;
  float ptMin;
  float ptMax;

  int etaBin;
  float etaMin;
  float etaMax;

  int phiBin;
  float phiMin;
  float phiMax;

  int chi2Bin;
  float chi2Min;
  float chi2Max;

  //the histos
  MonitorElement* muReco;
  
  // global muon
  std::vector<MonitorElement*> etaGlbTrack;
  std::vector<MonitorElement*> phiGlbTrack;
  std::vector<MonitorElement*> chi2OvDFGlbTrack;
  std::vector<MonitorElement*> ptGlbTrack;

  // tight muon
  MonitorElement* etaTight;
  MonitorElement* phiTight;
  MonitorElement* chi2OvDFTight;
  MonitorElement* ptTight;

  // tracker muon
  MonitorElement* etaTrack;
  MonitorElement* phiTrack;
  MonitorElement* chi2OvDFTrack;
  MonitorElement* ptTrack;
  // sta muon
  MonitorElement* etaStaTrack;
  MonitorElement* phiStaTrack;
  MonitorElement* chi2OvDFStaTrack;
  MonitorElement* ptStaTrack;

};
#endif
