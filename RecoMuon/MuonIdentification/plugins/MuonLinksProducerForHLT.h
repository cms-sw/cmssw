#ifndef MuonIdentification_MuonLinksProducerForHLT_h
#define MuonIdentification_MuonLinksProducerForHLT_h

/** \class MuonLinksProducerForHLT
 *
 * Simple producer to make reco::MuonTrackLinks collection 
 * out of the global muons from "muons" collection to restore
 * dropped links used as input for MuonIdProducer.
 *
 *  \author R. Bellan - UCSB <riccardo.bellan@cern.ch>
 */

// user include files
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class MuonLinksProducerForHLT : public edm::global::EDProducer<> {
public:
  explicit MuonLinksProducerForHLT(const edm::ParameterSet&);
  ~MuonLinksProducerForHLT() override = default;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::InputTag theLinkCollectionInInput_;
  const edm::InputTag theInclusiveTrackCollectionInInput_;
  const edm::EDGetTokenT<reco::MuonTrackLinksCollection> linkToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const double ptMin_;
  const double pMin_;
  const double shareHitFraction_;
};
#endif
