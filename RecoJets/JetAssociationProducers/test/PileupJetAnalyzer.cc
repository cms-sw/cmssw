#include <cmath>
#include <map>

#include <TNtuple.h>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

using namespace reco;

class PileupJetAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  PileupJetAnalyzer(const edm::ParameterSet &params);
  ~PileupJetAnalyzer();

  virtual void analyze(const edm::Event &event, const edm::EventSetup &es);

private:
  TNtuple *ntuple;

  edm::EDGetTokenT<JetTracksAssociationCollection> jetTracksAssocToken;
  edm::EDGetTokenT<JetFloatAssociation::Container> jetTagToken;
  edm::EDGetTokenT<edm::SimTrackContainer> tokenSimTrack;
  edm::EDGetTokenT<edm::SimVertexContainer> tokenSimVertex;
  const double signalFraction;
  const double jetMinE;
  const double jetMinEt;
  const double jetMaxEta;
  const double trackPtLimit;
};

PileupJetAnalyzer::PileupJetAnalyzer(const edm::ParameterSet &params)
    : signalFraction(params.getParameter<double>("signalFraction")),
      jetMinE(params.getParameter<double>("jetMinE")),
      jetMinEt(params.getParameter<double>("jetMinEt")),
      jetMaxEta(params.getParameter<double>("jetMaxEta")),
      trackPtLimit(params.getParameter<double>("trackPtLimit")) {
  jetTracksAssocToken = consumes<JetTracksAssociationCollection>(params.getParameter<edm::InputTag>("jetTracksAssoc"));
  jetTagToken = consumes<JetFloatAssociation::Container>(params.getParameter<edm::InputTag>("jetTagLabel"));
  tokenSimTrack = consumes<edm::SimTrackContainer>(edm::InputTag("fastSimProducer"));
  tokenSimVertex = consumes<edm::SimVertexContainer>(edm::InputTag("fastSimProducer"));

  usesResource(TFileService::kSharedResource);
  edm::Service<TFileService> fs;
  ntuple = fs->make<TNtuple>("jets", "", "mc:tag:et:eta");
}

PileupJetAnalyzer::~PileupJetAnalyzer() {}

void PileupJetAnalyzer::analyze(const edm::Event &event, const edm::EventSetup &es) {
  edm::Handle<JetTracksAssociationCollection> jetTracksAssoc;
  event.getByToken(jetTracksAssocToken, jetTracksAssoc);

  edm::Handle<JetFloatAssociation::Container> jetTags;
  event.getByToken(jetTagToken, jetTags);

  edm::Handle<edm::SimTrackContainer> simTracks;
  event.getByToken(tokenSimTrack, simTracks);

  edm::Handle<edm::SimVertexContainer> simVertices;
  event.getByToken(tokenSimVertex, simVertices);

  // find vertices in all jets
  for (JetTracksAssociationCollection::const_iterator iter = jetTracksAssoc->begin(); iter != jetTracksAssoc->end();
       ++iter) {
    const edm::RefToBase<Jet> &jet = iter->first;
    if (jet->energy() < jetMinE || jet->et() < jetMinEt || std::abs(jet->eta()) > jetMaxEta)
      continue;

    double tag = (*jetTags)[jet];

    double sigSum = 0.;
    double bkgSum = 0.;

    // note: the following code is FastSim specific right now

    const TrackRefVector &tracks = iter->second;
    for (TrackRefVector::const_iterator track = tracks.begin(); track != tracks.end(); ++track) {
      TrackingRecHitRef hitRef = (*track)->recHit(0);
      bool signal = false;
      unsigned int trackId;

      const FastTrackerRecHit *hit = dynamic_cast<const FastTrackerRecHit *>(&*hitRef);

      if (!hit)
        continue;
      trackId = hit->simTrackId(0);

      for (;;) {
        const SimTrack &simTrack = (*simTracks)[trackId];
        int genPartIndex = simTrack.genpartIndex();
        if (genPartIndex >= 0) {
          signal = true;
          break;
        }

        int vertIndex = simTrack.vertIndex();
        if (vertIndex < 0)
          break;

        const SimVertex &simVertex = (*simVertices)[vertIndex];
        int parentIndex = simVertex.parentIndex();
        if (parentIndex < 0)
          break;

        trackId = (unsigned int)parentIndex;
      }

      *(signal ? &sigSum : &bkgSum) += std::min((*track)->pt(), trackPtLimit);
    }

    if (sigSum + bkgSum < 1.0e-9)
      continue;
    double signal = sigSum / (sigSum + bkgSum);

    ntuple->Fill(signal, tag, jet->et(), jet->eta());
  }
}

DEFINE_FWK_MODULE(PileupJetAnalyzer);
