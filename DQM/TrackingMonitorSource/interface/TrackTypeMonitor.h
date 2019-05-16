#ifndef DQM_TrackingMonitorSource_TrackTypeMonitor_h
#define DQM_TrackingMonitorSource_TrackTypeMonitor_h

#include <string>
#include <vector>
#include <map>
#include <set>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

class TrackTypeMonitor : public DQMEDAnalyzer {
public:
  TrackTypeMonitor( const edm::ParameterSet& );

protected:

  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:

  void fillHistograms(const reco::Track& track, int indx);

  edm::ParameterSet parameters_;

  std::string moduleName_;
  std::string folderName_;
  bool verbose_;

  const edm::InputTag muonTag_;
  const edm::InputTag electronTag_;
  const edm::InputTag trackTag_;
  const edm::InputTag bsTag_;
  const edm::InputTag vertexTag_;

  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_; 
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;

  const std::string trackQuality_;
    
  std::vector<MonitorElement*> trackEtaHList_;
  std::vector<MonitorElement*> trackPhiHList_;
  std::vector<MonitorElement*> trackPHList_;
  std::vector<MonitorElement*> trackPtHList_;
  std::vector<MonitorElement*> trackPterrHList_;
  std::vector<MonitorElement*> trackqOverpHList_;
  std::vector<MonitorElement*> trackChi2bynDOFHList_;
  std::vector<MonitorElement*> nTracksHList_;
  std::vector<MonitorElement*> trackdzHList_;

  MonitorElement* hcounterH_;
  MonitorElement* dphiH_; 
  MonitorElement* drH_; 

  unsigned long long m_cacheID_;
};
#endif
