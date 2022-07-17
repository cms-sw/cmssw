#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"

class IsolationProducerForTracks : public edm::global::EDProducer<> {
public:
  IsolationProducerForTracks(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const override;

  const edm::EDGetTokenT<reco::CandidateView> tracksToken_;
  const edm::EDGetTokenT<reco::CandidateView> highPtTracksToken_;
  const edm::EDGetTokenT<reco::IsoDepositMap> isoDepsToken_;
  const double trackPtMin_;
  const double coneSize_;
};

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include <iostream>
#include <iterator>
#include <vector>

using namespace edm;
using namespace reco;

typedef edm::ValueMap<float> TkIsoMap;

IsolationProducerForTracks::IsolationProducerForTracks(const ParameterSet& pset)
    : tracksToken_(consumes<CandidateView>(pset.getParameter<InputTag>("tracks"))),
      highPtTracksToken_(consumes<CandidateView>(pset.getParameter<InputTag>("highPtTracks"))),
      isoDepsToken_(consumes<IsoDepositMap>(pset.getParameter<InputTag>("isoDeps"))),
      trackPtMin_(pset.getParameter<double>("trackPtMin")),
      coneSize_(pset.getParameter<double>("coneSize")) {
  produces<TkIsoMap>();
}

void IsolationProducerForTracks::produce(edm::StreamID, Event& event, const EventSetup& setup) const {
  auto caloIsolations = std::make_unique<TkIsoMap>();
  TkIsoMap::Filler filler(*caloIsolations);
  {
    Handle<CandidateView> tracks;
    event.getByToken(tracksToken_, tracks);

    Handle<CandidateView> highPtTracks;
    event.getByToken(highPtTracksToken_, highPtTracks);

    Handle<IsoDepositMap> isoDeps;
    event.getByToken(isoDepsToken_, isoDeps);

    int nTracks = tracks->size();
    int nHighPtTracks = highPtTracks->size();
    std::vector<double> iso(nTracks);

    OverlapChecker overlap;

    for (int i = 0; i < nTracks; ++i) {
      const Candidate& tkCand = (*tracks)[i];
      double caloIso = -1.0;
      if (tkCand.pt() > trackPtMin_) {
        for (int j = 0; j < nHighPtTracks; ++j) {
          const Candidate& highPtTkCand = (*highPtTracks)[j];
          if (overlap(tkCand, highPtTkCand)) {
            CandidateBaseRef tkRef = highPtTracks->refAt(j);
            const IsoDeposit& isoDep = (*isoDeps)[tkRef];
            caloIso = isoDep.depositWithin(coneSize_);
            break;
          }
        }
      }
      iso[i] = caloIso;
    }
    filler.insert(tracks, iso.begin(), iso.end());
  }

  // really fill the association map
  filler.fill();
  event.put(std::move(caloIsolations));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(IsolationProducerForTracks);
