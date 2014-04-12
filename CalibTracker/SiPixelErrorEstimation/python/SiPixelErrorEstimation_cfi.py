import FWCore.ParameterSet.Config as cms

PixelErrorEstimation = cms.EDAnalyzer("SiPixelErrorEstimation",
    # The type of particle that the simHit associated with recHits should be
    genType = cms.int32(13),
    # Replace  "ctfWithMaterialTracks" with "generalTracks"
    #untracked string src = "ctfWithMaterialTracks"
    src = cms.untracked.string('generalTracks'),
    outputFile = cms.untracked.string('SiPixelErrorEstimation_Ntuple.root'),
    # Include track hits ?
    include_trk_hits = cms.bool(True),
    # Do we check that the simHit associated with recHit is of the expected particle type ?
    checkType = cms.bool(False)
)


