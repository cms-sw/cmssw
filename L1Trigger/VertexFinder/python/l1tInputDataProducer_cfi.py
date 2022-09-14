import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.l1tVertexNTupler_cfi import l1tVertexNTupler
from L1Trigger.VertexFinder.l1tVertexProducer_cfi import l1tVertexProducer

l1tInputDataProducer = cms.EDProducer('InputDataProducer',
  #=== The name of the output collection
  outputCollectionName = cms.string("InputData"),
  
  #=== Cuts on MC truth particles (i.e., tracking particles) used for tracking efficiency measurements.
  GenCuts = l1tVertexNTupler.GenCuts,

  #=== Rules for deciding when the track finding has found an L1 track candidate
  L1TrackDef = l1tVertexNTupler.L1TrackDef,

  #=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
  TrackMatchDef = l1tVertexNTupler.TrackMatchDef,

  #=== Vertex Reconstruction configuration
  VertexReconstruction = l1tVertexProducer.VertexReconstruction,

  #=== Input collections
  hepMCInputTag = cms.InputTag("generatorSmeared","","SIM"),
  genParticleInputTag = cms.InputTag("genParticles",""),
  tpInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  tpValueMapInputTag = cms.InputTag("l1tTPStubValueMapProducer:TPs"),
  stubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),

  #=== Debug printout
  debug = l1tVertexNTupler.debug,
)
