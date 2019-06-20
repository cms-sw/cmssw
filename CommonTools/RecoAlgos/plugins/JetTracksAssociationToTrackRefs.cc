#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include <unordered_set>

/**
 * The purpose of this producer is to convert
 * AssociationVector<JetRefBaseProd, vector<RefVector<Track> > to a
 * RefVector<Track> of the (unique) values.
 */
class JetTracksAssociationToTrackRefs : public edm::global::EDProducer<> {
public:
  JetTracksAssociationToTrackRefs(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  edm::EDGetTokenT<reco::JetTracksAssociation::Container> associationToken_;
  edm::EDGetTokenT<edm::View<reco::Jet>> jetToken_;
  edm::EDGetTokenT<reco::JetCorrector> correctorToken_;
  const double ptMin_;
};

JetTracksAssociationToTrackRefs::JetTracksAssociationToTrackRefs(const edm::ParameterSet& iConfig)
    : associationToken_(
          consumes<reco::JetTracksAssociation::Container>(iConfig.getParameter<edm::InputTag>("association"))),
      jetToken_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      correctorToken_(consumes<reco::JetCorrector>(iConfig.getParameter<edm::InputTag>("corrector"))),
      ptMin_(iConfig.getParameter<double>("correctedPtMin")) {
  produces<reco::TrackRefVector>();
}

void JetTracksAssociationToTrackRefs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("association", edm::InputTag("ak4JetTracksAssociatorAtVertexPF"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("corrector", edm::InputTag("ak4PFL1FastL2L3Corrector"));
  desc.add<double>("correctedPtMin", 0);
  descriptions.add("jetTracksAssociationToTrackRefs", desc);
}

void JetTracksAssociationToTrackRefs::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<reco::JetTracksAssociation::Container> h_assoc;
  iEvent.getByToken(associationToken_, h_assoc);
  const reco::JetTracksAssociation::Container& association = *h_assoc;

  edm::Handle<edm::View<reco::Jet>> h_jets;
  iEvent.getByToken(jetToken_, h_jets);
  const edm::View<reco::Jet>& jets = *h_jets;

  edm::Handle<reco::JetCorrector> h_corrector;
  iEvent.getByToken(correctorToken_, h_corrector);
  const reco::JetCorrector& corrector = *h_corrector;

  auto ret = std::make_unique<reco::TrackRefVector>();
  std::unordered_set<reco::TrackRefVector::key_type> alreadyAdded;

  // Exctract tracks only for jets passing certain pT threshold after
  // correction
  for (size_t i = 0; i < jets.size(); ++i) {
    edm::RefToBase<reco::Jet> jetRef = jets.refAt(i);
    const reco::Jet& jet = *jetRef;

    auto p4 = jet.p4();

    // Energy correction in the most general way
    if (!corrector.vectorialCorrection()) {
      double scale = 1;
      if (!corrector.refRequired()) {
        scale = corrector.correction(jet);
      } else {
        scale = corrector.correction(jet, jetRef);
      }
      p4 = p4 * scale;
    } else {
      corrector.correction(jet, jetRef, p4);
    }

    if (p4.pt() <= ptMin_)
      continue;

    for (const auto& trackRef : association[jetRef]) {
      if (alreadyAdded.find(trackRef.key()) == alreadyAdded.end()) {
        ret->push_back(trackRef);
        alreadyAdded.insert(trackRef.key());
      }
    }
  }

  iEvent.put(std::move(ret));
}

DEFINE_FWK_MODULE(JetTracksAssociationToTrackRefs);
