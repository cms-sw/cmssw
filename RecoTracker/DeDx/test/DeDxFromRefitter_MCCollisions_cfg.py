import FWCore.ParameterSet.Config as cms

process = cms.Process("REFITTING")
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryPilot2_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.RawToDigi_cff")

#process.load("Configuration.StandardSequences.Digi_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("Configuration.EventContent.EventContent_cff")

process.load("RecoTracker.DeDx.dedxEstimatorsFromRefitter_cff")

#process.load("RecoTracker.DeDx.dedxEstimators_cff")
#process.dedxTruncated40.tracks=cms.InputTag("TrackRefitter")
#process.dedxTruncated40.trajectoryTrackAssociation = cms.InputTag("TrackRefitter")
#process.dedxHarmonic2.tracks=cms.InputTag("TrackRefitter")
#process.dedxHarmonic2.trajectoryTrackAssociation = cms.InputTag("TrackRefitter")

#process.dedxMedian.tracks=cms.InputTag("TrackRefitter")
#process.dedxMedian.trajectoryTrackAssociation = cms.InputTag("TrackRefitter")

#process.load("RecoTracker.TrackProducer.TrackRefitter_cff")
#process.TrackRefitter.TrajectoryInEvent = True


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    catalog = cms.untracked.string('PoolFileCatalog.xml'),
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v1/0001/0488E703-C27D-DD11-9F5E-001617C3B710.root')
    fileNames = cms.untracked.vstring('file:/tmp/gbruno/rereco.root')
                            
)

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck", ignoreTotal = cms.untracked.int32(1) )
#process.ProfilerService = cms.Service("SimpleMemoryCheck", firstEvent = cms.untracked.int32(2), lastEvent = cms.untracked.int32(10), paths = cms.untracked.vstring('p1') )


#RECO content
process.RECOAndDeDxEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
'keep *'))
#recoTracks_TrackRefitterTracks_*_*', 
#        'keep recoTrackExtras_TrackRefitterTracks_*_*', 
#        'keep TrackingRecHitsOwned_TrackRefitterTracks_*_*', 
#        'keep recoTracks_generalTracks_*_*', 
#        'keep recoTrackExtras_generalTracks_*_*', 
#        'keep TrackingRecHitsOwned_generalTracks_*_*', 
#        'keep recoTracks_rsWithMaterialTracks_*_*', 
#        'keep recoTrackExtras_rsWithMaterialTracks_*_*', 
#        'keep TrackingRecHitsOwned_rsWithMaterialTracks_*_*' ,'keep *_dedxHarmonic2_*_*' ,'keep *_dedxMedian_*_*', 'keep *_dedxTruncated40_*_*' )
#)

#process.RECOAndDeDxEventContent.outputCommands.extend(process.RECOEventContent.outputCommands)



process.RE_RECO = cms.OutputModule("PoolOutputModule",
    process.RECOAndDeDxEventContent,
    fileName = cms.untracked.string('/tmp/gbruno/rerecorefitter.root')
)

#process.p1 = cms.Path(process.reconstruction * (process.doAlldEdXEstimators + process.doAlldEdXDiscriminators) )
#process.p1 = cms.Path(process.TrackRefitter * process.doAlldEdXEstimators )
process.p1 = cms.Path( process.doAlldEdXEstimators )
process.outpath = cms.EndPath(process.RE_RECO)
process.GlobalTag.globaltag = "IDEAL_V5::All"

