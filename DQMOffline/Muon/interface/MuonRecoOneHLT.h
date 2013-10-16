#ifndef MuonRecoOneHLT_H
#define MuonRecoOneHLT_H


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class MuonRecoOneHLT : public edm::EDAnalyzer { //MuonAnalyzerBase {
 public:

  /// Constructor
  MuonRecoOneHLT(const edm::ParameterSet&); //, MuonServiceProxy *theService);
  
  /// Destructor
  ~MuonRecoOneHLT();

  /// Inizialize parameters for histo binning
  void beginJob();
  void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);
  
  /// Get the analysis
  //  void analyze(const edm::Event&, const edm::EventSetup&, const reco::Muon&, const edm::TriggerResults&);
  //  void analyze(const edm::Event&, const edm::EventSetup&, const edm::TriggerResults&, const reco::Vertex&);
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  // ----------member data ---------------------------
  DQMStore* dbe;
  edm::ParameterSet parameters;
  MuonServiceProxy *theService;

  // Switch for verbosity
  std::string metname;

  // STA Label
  edm::EDGetTokenT<reco::MuonCollection>   theMuonCollectionLabel_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot>         theBeamSpotLabel_;
  edm::EDGetTokenT<edm::TriggerResults>    theTriggerResultsLabel_;
    
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
