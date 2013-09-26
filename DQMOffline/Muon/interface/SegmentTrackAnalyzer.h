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
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/SegmentsTrackAssociator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class MuonServiceProxy;

class SegmentTrackAnalyzer : public edm::EDAnalyzer {
 public:

  /// Constructor
  SegmentTrackAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~SegmentTrackAnalyzer() {};
  
  /// Inizialize parameters for histo binning
  void beginJob();
  void beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup);

  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  // ----------member data ---------------------------
  DQMStore *theDbe;
  MuonServiceProxy *theService;
  edm::ParameterSet parameters;
  edm::EDGetTokenT<reco::TrackCollection> theMuTrackCollectionLabel_;
  
  // Switch for verbosity
  std::string metname;
  // Find the segments associated to the track
  SegmentsTrackAssociator* theSegmentsAssociator;

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
