/** \class GlobalMuonToMuonProducer
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoMuon/MuonIdentification/plugins/GlobalMuonToMuonProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// tmp
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

/// Constructor
GlobalMuonToMuonProducer::GlobalMuonToMuonProducer(const edm::ParameterSet& pSet) {
  theLinksCollectionLabel = pSet.getParameter<edm::InputTag>("InputObjects");

  setAlias(pSet.getParameter<std::string>("@module_label"));
  produces<reco::MuonCollection>().setBranchAlias(theAlias + "s");
  trackLinkToken_ = consumes<reco::MuonTrackLinksCollection>(theLinksCollectionLabel);
  trackingGeomToken_ = esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>();
}

/// Destructor
GlobalMuonToMuonProducer::~GlobalMuonToMuonProducer() {}

void GlobalMuonToMuonProducer::printTrackRecHits(const reco::Track& track,
                                                 edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const {
  const std::string metname = "Muon|RecoMuon|MuonIdentification|GlobalMuonToMuonProducer";

  LogTrace(metname) << "Valid RecHits: " << track.found() << " invalid RecHits: " << track.lost();

  int i = 0;
  for (trackingRecHit_iterator recHit = track.recHitsBegin(); recHit != track.recHitsEnd(); ++recHit)
    if ((*recHit)->isValid()) {
      const GeomDet* geomDet = trackingGeometry->idToDet((*recHit)->geographicalId());
      double r = geomDet->surface().position().perp();
      double z = geomDet->toGlobal((*recHit)->localPosition()).z();
      LogTrace(metname) << i++ << " r: " << r << " z: " << z << " " << geomDet->toGlobal((*recHit)->localPosition())
                        << std::endl;
    }
}

/// reconstruct muons
void GlobalMuonToMuonProducer::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& eventSetup) const {
  const std::string metname = "Muon|RecoMuon|MuonIdentification|GlobalMuonToMuonProducer";

  // the muon collection, it will be loaded in the event
  auto muonCollection = std::make_unique<reco::MuonCollection>();

  edm::Handle<reco::MuonTrackLinksCollection> linksCollection;
  event.getByToken(trackLinkToken_, linksCollection);

  if (linksCollection->empty()) {
    event.put(std::move(muonCollection));
    return;
  }

  // Global Tracking Geometry
  edm::ESHandle<GlobalTrackingGeometry> trackingGeometry = eventSetup.getHandle(trackingGeomToken_);

  for (reco::MuonTrackLinksCollection::const_iterator links = linksCollection->begin(); links != linksCollection->end();
       ++links) {
    // some temporary print-out
    LogTrace(metname) << "trackerTrack";
    printTrackRecHits(*(links->trackerTrack()), trackingGeometry);
    LogTrace(metname) << "standAloneTrack";
    printTrackRecHits(*(links->standAloneTrack()), trackingGeometry);
    LogTrace(metname) << "globalTrack";
    printTrackRecHits(*(links->globalTrack()), trackingGeometry);

    // Fill the muon
    reco::Muon muon;
    muon.setStandAlone(links->standAloneTrack());
    muon.setTrack(links->trackerTrack());
    muon.setCombined(links->globalTrack());

    // FIXME: can this break in case combined info cannot be added to some tracks?
    muon.setCharge(links->globalTrack()->charge());

    //FIXME: E = sqrt(p^2 + m^2), where m == 0.105658369(9)GeV
    double energy = sqrt(links->globalTrack()->p() * links->globalTrack()->p() + 0.011163691);
    math::XYZTLorentzVector p4(
        links->globalTrack()->px(), links->globalTrack()->py(), links->globalTrack()->pz(), energy);

    muon.setP4(p4);
    muon.setVertex(links->globalTrack()->vertex());

    muonCollection->push_back(muon);
  }

  event.put(std::move(muonCollection));
}
