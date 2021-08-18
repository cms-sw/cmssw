import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexNTupler_cff import L1TVertexNTupler
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer

InputDataProducer = cms.EDProducer('InputDataProducer',
  #=== The name of the output collection
  outputCollectionName = cms.string("InputData"),
  
  #=== Cuts on MC truth particles (i.e., tracking particles) used for tracking efficiency measurements.
  GenCuts = L1TVertexNTupler.GenCuts,

  #=== Rules for deciding when the track finding has found an L1 track candidate
  L1TrackDef = L1TVertexNTupler.L1TrackDef,

  #=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
  TrackMatchDef = L1TVertexNTupler.TrackMatchDef,

  #=== Vertex Reconstruction configuration
  VertexReconstruction = VertexProducer.VertexReconstruction,

  #=== Input collections
  hepMCInputTag = cms.InputTag("generatorSmeared","","SIM"),
  genParticleInputTag = cms.InputTag("genParticles",""),
  tpInputTag = cms.InputTag("mix", "MergedTrackTruth"),
  tpValueMapInputTag = cms.InputTag("TPStubValueMapProducer:TPs"),
  stubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis", "StubAccepted"),

  #=== Debug printout
  debug = L1TVertexNTupler.debug,
)