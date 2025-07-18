/** \class MuonLinksProducerForHLT
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoMuon/MuonIdentification/plugins/MuonLinksProducerForHLT.h"

MuonLinksProducerForHLT::MuonLinksProducerForHLT(const edm::ParameterSet& iConfig)
    : theLinkCollectionInInput_{iConfig.getParameter<edm::InputTag>("LinkCollection")},
      theInclusiveTrackCollectionInInput_{iConfig.getParameter<edm::InputTag>("InclusiveTrackerTrackCollection")},
      linkToken_{consumes<reco::MuonTrackLinksCollection>(theLinkCollectionInInput_)},
      trackToken_{consumes<reco::TrackCollection>(theInclusiveTrackCollectionInInput_)},
      ptMin_{iConfig.getParameter<double>("ptMin")},
      pMin_{iConfig.getParameter<double>("pMin")},
      shareHitFraction_{iConfig.getParameter<double>("shareHitFraction")} {
  produces<reco::MuonTrackLinksCollection>();
}

void MuonLinksProducerForHLT::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto output = std::make_unique<reco::MuonTrackLinksCollection>();

  edm::Handle<reco::MuonTrackLinksCollection> links;
  iEvent.getByToken(linkToken_, links);

  edm::Handle<reco::TrackCollection> incTracks;
  iEvent.getByToken(trackToken_, incTracks);

  for (reco::MuonTrackLinksCollection::const_iterator link = links->begin(); link != links->end(); ++link) {
    bool found = false;
    unsigned int trackIndex = 0;
    unsigned int muonTrackHits = link->trackerTrack()->extra()->recHitsSize();
    for (reco::TrackCollection::const_iterator track = incTracks->begin(); track != incTracks->end();
         ++track, ++trackIndex) {
      if (track->pt() < ptMin_)
        continue;
      if (track->p() < pMin_)
        continue;
      //std::cout << "pt (muon/track) " << link->trackerTrack()->pt() << " " << track->pt() << std::endl;
      unsigned trackHits = track->extra()->recHitsSize();
      //std::cout << "hits (muon/track) " << muonTrackHits  << " " << trackHits() << std::endl;
      unsigned int smallestNumberOfHits = trackHits < muonTrackHits ? trackHits : muonTrackHits;
      int numberOfCommonDetIds = 0;
      for (auto hit = track->extra()->recHitsBegin(); hit != track->extra()->recHitsEnd(); ++hit) {
        for (auto mit = link->trackerTrack()->extra()->recHitsBegin();
             mit != link->trackerTrack()->extra()->recHitsEnd();
             ++mit) {
          if ((*hit)->geographicalId() == (*mit)->geographicalId() &&
              (*hit)->sharesInput((*mit), TrackingRecHit::some)) {
            numberOfCommonDetIds++;
            break;
          }
        }
      }
      double fraction = (double)numberOfCommonDetIds / smallestNumberOfHits;
      // std::cout << "Overlap/Smallest/fraction = " << numberOfCommonDetIds << " " << smallestNumberOfHits << " " << fraction << std::endl;
      if (fraction > shareHitFraction_) {
        output->push_back(
            reco::MuonTrackLinks(reco::TrackRef(incTracks, trackIndex), link->standAloneTrack(), link->globalTrack()));
        found = true;
        break;
      }
    }
    if (!found)
      output->push_back(reco::MuonTrackLinks(link->trackerTrack(), link->standAloneTrack(), link->globalTrack()));
  }
  iEvent.put(std::move(output));
}

void MuonLinksProducerForHLT::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("LinkCollection", edm::InputTag("hltPFMuonMerging"));
  desc.add<edm::InputTag>("InclusiveTrackerTrackCollection", edm::InputTag("hltL3MuonsLinksCombination"));
  desc.add<double>("ptMin", 2.5);
  desc.add<double>("pMin", 2.5);
  desc.add<double>("shareHitFraction", 0.80);
  descriptions.addWithDefaultLabel(desc);
}
