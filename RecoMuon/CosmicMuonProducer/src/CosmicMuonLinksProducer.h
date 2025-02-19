#ifndef RecoMuon_CosmicMuonProducer_CosmicMuonLinksProducer_H
#define RecoMuon_CosmicMuonProducer_CosmicMuonLinksProducer_H

/** \file CosmicMuonLinksProducer
 *
 *  $Date: 2010/07/19 19:54:15 $
 *  $Revision: 1.2 $
 *  \author Chang Liu - Purdue University <chang.liu@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"

class MuonServiceProxy;

class CosmicMuonLinksProducer : public edm::EDProducer {
public:
  explicit CosmicMuonLinksProducer(const edm::ParameterSet&);

   ~CosmicMuonLinksProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

private:

  reco::TrackToTrackMap mapTracks(const edm::Handle<reco::TrackCollection>&, const edm::Handle<reco::TrackCollection>&) const;

  int sharedHits(const reco::Track& track1, const reco::Track& track2) const;

  MuonServiceProxy* theService;

  std::vector<std::pair<edm::InputTag, edm::InputTag> > theTrackLinks;

  std::string category_;

};

#endif
