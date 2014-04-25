#ifndef RecoMuon_CosmicMuonProducer_GlobalCosmicMuonProducer_H
#define RecoMuon_CosmicMuonProducer_GlobalCosmicMuonProducer_H

/** \file CosmicMuonProducer
 *
 *  reconstruct muons using dt,csc,rpc and tracker starting from cosmic muon 
 *  tracks
 *
 *  \author Chang Liu  -  Purdue University <Chang.Liu@cern.ch>
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

class MuonTrackFinder;
class MuonServiceProxy;

class GlobalCosmicMuonProducer : public edm::EDProducer {
public:
  explicit GlobalCosmicMuonProducer(const edm::ParameterSet&);

   ~GlobalCosmicMuonProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<reco::TrackCollection> theTrackCollectionToken;
  MuonTrackFinder* theTrackFinder;

  /// the event setup proxy, it takes care the services update
  MuonServiceProxy *theService;
};

#endif
