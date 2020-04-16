#ifndef RecoBTag_SoftLepton_SoftPFElectronTagInfoProducer_h
#define RecoBTag_SoftLepton_SoftPFElectronTagInfoProducer_h

#include <vector>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

// Vertex
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
// SoftPFElectronTagInfoProducer:  the SoftPFElectronTagInfoProducer takes
// a PFCandidateCollection as input and produces a RefVector
// to the likely soft electrons in this collection.

class SoftPFElectronTagInfoProducer : public edm::stream::EDProducer<> {
public:
  SoftPFElectronTagInfoProducer(const edm::ParameterSet& conf);
  ~SoftPFElectronTagInfoProducer() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  bool isElecClean(edm::Event&, const reco::GsfElectron*);
  float boostedPPar(const math::XYZVector&, const math::XYZVector&);

  // service used to make transient tracks from tracks
  const TransientTrackBuilder* transientTrackBuilder;
  edm::EDGetTokenT<reco::VertexCollection> token_primaryVertex;
  edm::EDGetTokenT<edm::View<reco::Jet> > token_jets;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > token_elec;
  edm::EDGetTokenT<reco::BeamSpot> token_BeamSpot;
  edm::EDGetTokenT<reco::ConversionCollection> token_allConversions;
  float DeltaRElectronJet, MaxSip3Dsig;
  bool goodvertex;

  const reco::Vertex* vertex;
};

#endif
