#ifndef TriggerMatchMonitor_H
#define TriggerMatchMonitor_H
/** \class TriggerMatch monitor
 *
 *  DQM monitoring source for Trigger matching feature added to miniAOD
 *
 *  \author Bibhuprasad Mahakud (Purdue University, West Lafayette, USA)
 */

#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"

class TriggerMatchMonitor : public DQMEDAnalyzer {
public:
  /// Constructor
  TriggerMatchMonitor(const edm::ParameterSet& pSet);

  /// Destructor
  ~TriggerMatchMonitor() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------
  edm::ParameterSet parameters;

  // triggerNames to be passed from config
  std::string triggerPathName1_;
  std::string triggerHistName1_;
  double triggerPtThresholdPath1_;
  std::string triggerPathName2_;
  std::string triggerHistName2_;
  double triggerPtThresholdPath2_;

  //Vertex requirements
  edm::EDGetTokenT<edm::View<reco::Muon>> theMuonCollectionLabel_;
  edm::EDGetTokenT<edm::View<pat::Muon>> thePATMuonCollectionLabel_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotLabel_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<std::vector<pat::TriggerObjectStandAlone>> triggerObjects_;

  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> primaryVerticesToken_;

  // histograms
  std::vector<MonitorElement*> matchHists;
  MonitorElement* h_passHLTPath1_eta_Tight;
  MonitorElement* h_passHLTPath1_pt_Tight;
  MonitorElement* h_passHLTPath1_phi_Tight;
  MonitorElement* h_totalHLTPath1_eta_Tight;
  MonitorElement* h_totalHLTPath1_pt_Tight;
  MonitorElement* h_totalHLTPath1_phi_Tight;

  MonitorElement* h_passHLTPath2_eta_Tight;
  MonitorElement* h_passHLTPath2_pt_Tight;
  MonitorElement* h_passHLTPath2_phi_Tight;
  MonitorElement* h_totalHLTPath2_eta_Tight;
  MonitorElement* h_totalHLTPath2_pt_Tight;
  MonitorElement* h_totalHLTPath2_phi_Tight;

  std::string theFolder;
};
#endif
