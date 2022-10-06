#ifndef JetSignalVertexCompatibility_h
#define JetSignalVertexCompatibility_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "RecoJets/JetAssociationAlgorithms/interface/JetSignalVertexCompatibilityAlgo.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"

class TransientTrackBuilder;
class TransientTrackRecord;

class JetSignalVertexCompatibility : public edm::stream::EDProducer<> {
public:
  JetSignalVertexCompatibility(const edm::ParameterSet &params);
  ~JetSignalVertexCompatibility() override;

  void produce(edm::Event &event, const edm::EventSetup &es) override;

private:
  reco::JetSignalVertexCompatibilityAlgo algo;

  edm::EDGetTokenT<reco::JetTracksAssociationCollection> jetTracksAssocToken;
  edm::EDGetTokenT<reco::VertexCollection> primaryVerticesToken;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> builderToken;
};

#endif  // JetSignalVertexCompatibility_h
