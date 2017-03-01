import FWCore.ParameterSet.Config as cms

process = cms.Process("TrackAlgoCompare")

# keep message logger to a nice level
process.MessageLogger = cms.Service("MessageLogger",
    default = cms.untracked.PSet(
        limit = cms.untracked.int32(10)
    )
)

### input files
process.source = cms.Source("PoolSource", 
    #fileNames = cms.untracked.vstring('file:${HOME}/samples/ttbar.root') 
    fileNames = cms.untracked.vstring('file:${HOME}/samples/ttbar2.root') 
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

# Track Associators
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")

process.TrackAssociatorByHits.SimToRecoDenominator = 'reco'

# Include MagneticField Record (note: you must have this module included for the association to work)
process.load("Configuration.StandardSequences.MagneticField_cff")

# load filters for recoTracks and trackingParticles 
process.load("RecoTracker.DebugTools.cuts_cff")

# filter for recoTracks algo A and B
#process.cutsRTAlgoA.src   = cms.InputTag("generalTracks")
#process.cutsRTAlgoA.ptMin = cms.double(1.0)
#process.cutsRTAlgoB.src   = cms.InputTag("generalTracks")
#process.cutsRTAlgoB.ptMin = cms.double(1.0)

# filter for trackingParticles efficiency and fakes
#process.cutsTPEffic.src   = cms.InputTag("mix","MergedTrackTruth")
#process.cutsTPEffic.ptMin = cms.double(0.1)
#process.cutsTPFake.src    = cms.InputTag("mix","MergedTrackTruth")
#process.cutsTPFake.ptMin  = cms.double(0.1)

# Include TrackAlgoCompareUtil cfi
process.load("RecoTracker.DebugTools.TrackAlgoCompareUtil_cff")
#process.trackAlgoCompareUtil.trackLabel_algoA = cms.InputTag("cutsRTAlgoA")
#process.trackAlgoCompareUtil.trackLabel_algoB = cms.InputTag("cutsRTAlgoB")
#process.trackAlgoCompareUtil.trackingParticleLabel_effic = cms.InputTag("cutsTPEffic")
#process.trackAlgoCompareUtil.trackingParticleLabel_fakes = cms.InputTag("cutsTPFake")
process.trackAlgoCompareUtil.UseAssociators = cms.bool(True)
process.trackAlgoCompareUtil.assocLabel_algoA = cms.untracked.string("trackAssociationByHits"); 
process.trackAlgoCompareUtil.assocLabel_algoB = cms.untracked.string("trackAssociationByHits"); 

process.out = cms.OutputModule("PoolOutputModule", 
    outputCommands = cms.untracked.vstring(
        'drop *_*_*_*',
        'keep recoTracks_generalTracks_*_*',
        'keep recoTracks_cutsRTAlgoA_*_*',
        'keep recoTracks_cutsRTAlgoB_*_*',
        'keep TrackingParticles_mergedtruth_*_*',
        'keep TrackingParticles_cutsTPEffic_*_*',
        'keep TrackingParticles_cutsTPFake_*_*',
        'keep recoVertexs_offlinePrimaryVertices_*_*',
        'keep TrackingVertexs_mergedtruth_*_*',
        'keep recoBeamSpot_offlineBeamSpot_*_',
        'keep *_*_*_TrackAlgoCompare'
        ),
    fileName = cms.untracked.string('TrackAlgoCompareOutput.root')
)

process.p = cms.Path(process.trackAssociationByHits+process.trackAlgoCompareUtil)
#process.p = cms.Path(process.cutsRTAlgoA + process.cutsRTAlgoB + process.cutsTPEffic + process.cutsTPFake + process.trackAlgoCompareUtil)
process.ep = cms.EndPath(process.out)


