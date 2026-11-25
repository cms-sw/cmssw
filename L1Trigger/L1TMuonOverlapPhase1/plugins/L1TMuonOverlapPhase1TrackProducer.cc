#include "L1TMuonOverlapPhase1TrackProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLink.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <algorithm>
#include <iostream>
#include <memory>

L1TMuonOverlapPhase1TrackProducer::L1TMuonOverlapPhase1TrackProducer(const edm::ParameterSet& edmParameterSet)
    : muStubsInputTokens(
          {consumes<L1MuDTChambPhContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTPh")),
           consumes<L1MuDTChambThContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTTh")),
           consumes<CSCCorrelatedLCTDigiCollection>(edmParameterSet.getParameter<edm::InputTag>("srcCSC")),
           consumes<RPCDigiCollection>(edmParameterSet.getParameter<edm::InputTag>("srcRPC"))}),
      omtfParamsEsToken(esConsumes<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd, edm::Transition::BeginRun>()),
      muonGeometryTokens({esConsumes<RPCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>(),
                          esConsumes<CSCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>(),
                          esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>()}),
      magneticFieldEsToken(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagatorEsToken(esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", "SteppingHelixPropagatorAlong"))),
      //propagatorEsToken(esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(edmParameterSet.getParameter<edm::ESInputTag>("propagatorTag"))),
      omtfReconstruction(edmParameterSet, muStubsInputTokens) {
  produces<l1t::RegionalMuonCandBxCollection>("OMTF");

  if (edmParameterSet.exists("simTracksTag"))
    consumes<edm::SimTrackContainer>(edmParameterSet.getParameter<edm::InputTag>("simTracksTag"));
  if (edmParameterSet.exists("simVertexesTag"))
    consumes<edm::SimVertexContainer>(edmParameterSet.getParameter<edm::InputTag>("simVertexesTag"));
  if (edmParameterSet.exists("trackingParticleTag"))
    consumes<TrackingParticleCollection>(edmParameterSet.getParameter<edm::InputTag>("trackingParticleTag"));

  if (edmParameterSet.exists("genParticleTag"))
    consumes<reco::GenParticleCollection>(edmParameterSet.getParameter<edm::InputTag>("genParticleTag"));

  if (edmParameterSet.exists("rpcSimHitsInputTag"))
    consumes<edm::PSimHitContainer>(edmParameterSet.getParameter<edm::InputTag>("rpcSimHitsInputTag"));
  if (edmParameterSet.exists("cscSimHitsInputTag"))
    consumes<edm::PSimHitContainer>(edmParameterSet.getParameter<edm::InputTag>("cscSimHitsInputTag"));
  if (edmParameterSet.exists("dtSimHitsInputTag"))
    consumes<edm::PSimHitContainer>(edmParameterSet.getParameter<edm::InputTag>("dtSimHitsInputTag"));

  if (edmParameterSet.exists("rpcDigiSimLinkInputTag"))
    consumes<edm::DetSetVector<RPCDigiSimLink> >(edmParameterSet.getParameter<edm::InputTag>("rpcDigiSimLinkInputTag"));
  if (edmParameterSet.exists("cscStripDigiSimLinksInputTag"))
    consumes<edm::DetSetVector<StripDigiSimLink> >(
        edmParameterSet.getParameter<edm::InputTag>("cscStripDigiSimLinksInputTag"));
  if (edmParameterSet.exists("dtDigiSimLinksInputTag"))
    consumes<MuonDigiCollection<DTLayerId, DTDigiSimLink> >(
        edmParameterSet.getParameter<edm::InputTag>("dtDigiSimLinksInputTag"));
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
L1TMuonOverlapPhase1TrackProducer::~L1TMuonOverlapPhase1TrackProducer() {}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase1TrackProducer::beginJob() { omtfReconstruction.beginJob(); }
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase1TrackProducer::endJob() { omtfReconstruction.endJob(); }
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase1TrackProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  omtfReconstruction.beginRun(
      run, iSetup, omtfParamsEsToken, muonGeometryTokens, magneticFieldEsToken, propagatorEsToken);
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase1TrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup) {
  std::ostringstream str;

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> candidates = omtfReconstruction.reconstruct(iEvent, evSetup);

  iEvent.put(std::move(candidates), "OMTF");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonOverlapPhase1TrackProducer);
