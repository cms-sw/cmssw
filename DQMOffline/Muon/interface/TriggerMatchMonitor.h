#ifndef TriggerMatchMonitor_H
#define TriggerMatchMonitor_H
/** \class MuRecoAnalyzer
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
#include "DQMServices/Core/interface/MonitorElement.h"
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
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
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
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:
  
  // ----------member data ---------------------------
  MuonServiceProxy *theService;
  DQMStore* theDbe;
  edm::ParameterSet parameters;
 
  // Switch for verbosity
  std::string metname;

  //Vertex requirements
  edm::EDGetTokenT<edm::View<reco::Muon> >   theMuonCollectionLabel_;
  edm::EDGetTokenT<edm::View<pat::Muon> >   thePATMuonCollectionLabel_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot>         theBeamSpotLabel_;
  edm::EDGetTokenT<edm::TriggerResults > triggerResultsToken_;
  edm::EDGetTokenT<std::vector<pat::TriggerObjectStandAlone>> triggerObjects_;

  edm::EDGetTokenT<reco::BeamSpot > beamSpotToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > primaryVerticesToken_;



  // global muon
  std::vector<MonitorElement*> matchHists;

  std::string theFolder;
};
#endif
