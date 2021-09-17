import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer

L1TVertexNTupler = cms.EDAnalyzer('VertexNTupler',
  inputDataInputTag = cms.InputTag("InputDataProducer","InputData"),
  genParticleInputTag = cms.InputTag("genParticles",""),
  l1TracksInputTags    = cms.VInputTag( VertexProducer.l1TracksInputTag ),
  l1TracksTruthMapInputTags = cms.VInputTag( cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks") ),
  l1TracksTPInputTags = cms.InputTag("TPStubValueMapProducer:allMatchedTPs"),
  l1TracksTPValueMapInputTags = cms.InputTag("TPStubValueMapProducer:TPs"),
  l1TracksBranchNames  = cms.vstring('hybrid'),
  l1VertexInputTags   = cms.VInputTag( cms.InputTag("VertexProducer", VertexProducer.l1VertexCollectionName.value()) ),
  l1VertexTrackInputs = cms.vstring('hybrid'),
  l1VertexBranchNames = cms.vstring('FastHisto'),
  extraL1VertexInputTags = cms.VInputTag(),
  extraL1VertexDescriptions = cms.vstring(),

  genJetsInputTag = cms.InputTag("ak4GenJetsNoNu"),

  #=== Cuts on MC truth particles (i.e., tracking particles) used for tracking efficiency measurements.
  GenCuts = cms.PSet(
     GenMinPt         = cms.double(2.0),
     GenMaxAbsEta     = cms.double(2.4),
     GenMaxVertR      = cms.double(1.0), # Maximum distance of particle production vertex from centre of CMS.
     GenMaxVertZ      = cms.double(30.0),
     GenPdgIds        = cms.vuint32(), # Only particles with these PDG codes used for efficiency measurement.


     # Additional cut on MC truth tracks used for algorithmic tracking efficiency measurements.
     # You should usually set this equal to value of L1TrackDef.MinStubLayers below, unless L1TrackDef.MinPtToReduceLayers
     # is < 10000, in which case, set it equal to (L1TrackDef.MinStubLayers - 1).
     GenMinStubLayers = cms.uint32(4)
  ),


  #=== Rules for deciding when the track finding has found an L1 track candidate
  L1TrackDef = cms.PSet(
     UseLayerID           = cms.bool(True),
     # Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware).
     ReducedLayerID       = cms.bool(True)
  ),

  #=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
  TrackMatchDef = cms.PSet(
     #--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, set two of them to ultra loose).
     # Min. fraction of matched stubs relative to number of stubs on reco track.
     MinFracMatchStubsOnReco  = cms.double(-99.),
     # Min. fraction of matched stubs relative to number of stubs on tracking particle.
     MinFracMatchStubsOnTP    = cms.double(-99.),
     # Min. number of matched layers.
     MinNumMatchLayers        = cms.uint32(4),
     # Min. number of matched PS layers.
     MinNumMatchPSLayers      = cms.uint32(0),
     # Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
     StubMatchStrict          = cms.bool(False)
  ),


  # === Vertex Reconstruction configuration
  VertexReconstruction = VertexProducer.VertexReconstruction,

  # Debug printout
  debug  = VertexProducer.debug,
  printResults = cms.bool(False)
)
