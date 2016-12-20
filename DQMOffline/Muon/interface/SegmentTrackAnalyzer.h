#ifndef SegmentTrackAnalyzer_H
#define SegmentTrackAnalyzer_H


/** \class SegmentTrackAnalyzer
 *
 *  DQM monitoring source for segments associated to the muon track
 *
 *  \author G. Mila - INFN Torino
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
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/SegmentsTrackAssociator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class MuonServiceProxy;

class SegmentTrackAnalyzer : public DQMEDAnalyzer {
 public:

  /// Constructor
  SegmentTrackAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~SegmentTrackAnalyzer() {
    delete theService;
    delete theSegmentsAssociator;
  };
  
  void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:
  // ----------member data ---------------------------
  MuonServiceProxy *theService;
  edm::ParameterSet parameters;
  edm::EDGetTokenT<reco::TrackCollection> theMuTrackCollectionLabel_;
  
  // Switch for verbosity
  std::string metname;
  std::string trackCollection;
  // Find the segments associated to the track
  SegmentsTrackAssociator* theSegmentsAssociator;

  int    etaBin;
  double etaMin;
  double etaMax;
  int    phiBin;
  double phiMin;
  double phiMax;
  int     ptBin;
  double  ptMin;
  double  ptMax;
  
  // the histos
  MonitorElement* hitsNotUsed;
  MonitorElement* hitsNotUsedPercentual;
  MonitorElement* TrackSegm;
  MonitorElement* hitStaProvenance;
  MonitorElement* hitTkrProvenance;
  MonitorElement* trackHitPercentualVsEta;
  MonitorElement* trackHitPercentualVsPhi;
  MonitorElement* trackHitPercentualVsPt;
  MonitorElement* dtTrackHitPercentualVsEta;
  MonitorElement* dtTrackHitPercentualVsPhi;
  MonitorElement* dtTrackHitPercentualVsPt;
  MonitorElement* cscTrackHitPercentualVsEta;
  MonitorElement* cscTrackHitPercentualVsPhi;
  MonitorElement* cscTrackHitPercentualVsPt;

 };
#endif  
