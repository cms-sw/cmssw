#ifndef TrackProducer_h
#define TrackProducer_h

/** \class TrackProducer
 *  Produce Tracks from TrackCandidates
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

class TrackFromPFProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit TrackFromPFProducer(const edm::ParameterSet& iConfig);

  /// Implementation of produce method
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<std::vector<pat::PackedCandidate> > tokenPFCandidates_;
  edm::EDGetTokenT<std::vector<pat::PackedCandidate> > tokenPFCandidatesLostTracks_;
};

#endif
