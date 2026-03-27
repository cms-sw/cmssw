#include "L1Trigger/L1TMuonOverlapPhase2/plugins/L1TMuonOverlapPhase2TrackProducer.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"

#include <algorithm>
#include <iostream>
#include <memory>

L1TMuonOverlapPhase2TrackProducer::L1TMuonOverlapPhase2TrackProducer(const edm::ParameterSet& edmParameterSet)
    : muStubsInputTokens(
          {mayConsume<L1MuDTChambPhContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTPh")),
           mayConsume<L1MuDTChambThContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTTh")),
           consumes<CSCCorrelatedLCTDigiCollection>(edmParameterSet.getParameter<edm::InputTag>("srcCSC")),
           consumes<RPCDigiCollection>(edmParameterSet.getParameter<edm::InputTag>("srcRPC"))}),
      muStubsPhase2InputTokens(
          {consumes<L1Phase2MuDTPhContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTPhPhase2")),
           consumes<L1Phase2MuDTThContainer>(edmParameterSet.getParameter<edm::InputTag>("srcDTThPhase2"))}),
      omtfParamsEsToken(esConsumes<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd, edm::Transition::BeginRun>()),
      muonGeometryTokens({esConsumes<RPCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>(),
                          esConsumes<CSCGeometry, MuonGeometryRecord, edm::Transition::BeginRun>(),
                          esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>()}),
      //needed for pattern generation and RootDataDumper
      magneticFieldEsToken(esConsumes<MagneticField, IdealMagneticFieldRecord, edm::Transition::BeginRun>()),
      propagatorEsToken(esConsumes<Propagator, TrackingComponentsRecord, edm::Transition::BeginRun>(
          edm::ESInputTag("", "SteppingHelixPropagatorAlong"))),
      omtfEmulation(edmParameterSet, muStubsInputTokens, muStubsPhase2InputTokens) {
  produces<l1t::RegionalMuonCandBxCollection>("OMTF");  //phase-1 collection
  produces<l1t::SAMuonCollection>("OMTFconstr");
  produces<l1t::SAMuonCollection>("OMTFunconstr");

  //it is needed for pattern generation and RootDataDumper
  if (edmParameterSet.exists("simTracksTag"))
    mayConsume<edm::SimTrackContainer>(edmParameterSet.getParameter<edm::InputTag>("simTracksTag"));
  if (edmParameterSet.exists("simVertexesTag"))
    mayConsume<edm::SimVertexContainer>(edmParameterSet.getParameter<edm::InputTag>("simVertexesTag"));
  if (edmParameterSet.exists("trackingParticleTag"))
    mayConsume<TrackingParticleCollection>(edmParameterSet.getParameter<edm::InputTag>("trackingParticleTag"));
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase2TrackProducer::beginJob() { omtfEmulation.beginJob(); }
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase2TrackProducer::endJob() { omtfEmulation.endJob(); }
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase2TrackProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  omtfEmulation.beginRun(run, iSetup, omtfParamsEsToken, muonGeometryTokens, magneticFieldEsToken, propagatorEsToken);
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonOverlapPhase2TrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup) {
  std::ostringstream str;

  auto outptuCollections = omtfEmulation.run(iEvent, evSetup);

  iEvent.put(std::move(outptuCollections.regionalCandidates), "OMTF");
  iEvent.put(std::move(outptuCollections.constrSaMuons), "OMTFconstr");
  iEvent.put(std::move(outptuCollections.unConstrSaMuons), "OMTFunconstr");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonOverlapPhase2TrackProducer);
