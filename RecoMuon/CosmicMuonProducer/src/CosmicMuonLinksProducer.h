#ifndef RecoMuon_CosmicMuonProducer_CosmicMuonLinksProducer_H
#define RecoMuon_CosmicMuonProducer_CosmicMuonLinksProducer_H

/** \file CosmicMuonLinksProducer
 *
 *  \author Chang Liu - Purdue University <chang.liu@cern.ch>
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

class MuonServiceProxy;

class CosmicMuonLinksProducer : public edm::stream::EDProducer<> {
public:
  explicit CosmicMuonLinksProducer(const edm::ParameterSet&);

  ~CosmicMuonLinksProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  reco::TrackToTrackMap mapTracks(const edm::Handle<reco::TrackCollection>&,
                                  const edm::Handle<reco::TrackCollection>&) const;

  int sharedHits(const reco::Track& track1, const reco::Track& track2) const;

  MuonServiceProxy* theService;

  std::vector<std::pair<edm::EDGetTokenT<reco::TrackCollection>, edm::EDGetTokenT<reco::TrackCollection> > >
      theTrackLinks;
  std::vector<std::pair<std::string, std::string> > theTrackLinkNames;

  std::string category_;
};

#endif
