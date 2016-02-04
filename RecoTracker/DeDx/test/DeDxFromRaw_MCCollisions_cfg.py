import FWCore.ParameterSet.Config as cms

process = cms.Process("RP")
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryPilot2_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.RawToDigi_cff")

#process.load("Configuration.StandardSequences.Digi_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("Configuration.EventContent.EventContent_cff")


process.load("RecoTracker.DeDx.dedxEstimators_cff")
#process.load("RecoTracker.DeDx.dedxDiscriminators_cff")



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    catalog = cms.untracked.string('PoolFileCatalog.xml'),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP_V7_v1/0001/0488E703-C27D-DD11-9F5E-001617C3B710.root')
#    fileNames = cms.untracked.vstring('/store/relval/2008/6/20/RelVal-RelValTTbar-1213920853/0000/028CEBF9-A53E-DD11-BF35-00161757BF42.root')
                            
)

process.Timing = cms.Service("Timing")
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck", ignoreTotal = cms.untracked.int32(1) )
#process.ProfilerService = cms.Service("SimpleMemoryCheck", firstEvent = cms.untracked.int32(2), lastEvent = cms.untracked.int32(10), paths = cms.untracked.vstring('p1') )


#RECO content

process.RECOAndDeDxEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)



process.DeDxEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
       'keep *_dedxHarmonic2_*_*' ,'keep *_dedxMedian_*_*', 'keep *_dedxTruncated40_*_*' )
)

process.RECOAndDeDxEventContent.outputCommands.extend(process.RecoLocalTrackerRECO.outputCommands)

process.RECOAndDeDxEventContent.outputCommands.extend(process.TrackingToolsRECO.outputCommands)

process.RECOAndDeDxEventContent.outputCommands.extend(process.RecoTrackerRECO.outputCommands)

process.RECOAndDeDxEventContent.outputCommands.extend(process.BeamSpotRECO.outputCommands)

process.RECOAndDeDxEventContent.outputCommands.extend(process.DeDxEventContent.outputCommands)


process.RE_RECO = cms.OutputModule("PoolOutputModule",
    process.RECOAndDeDxEventContent,
    fileName = cms.untracked.string('/tmp/gbruno/rereco.root')
)

#process.p1 = cms.Path(process.reconstruction * (process.doAlldEdXEstimators + process.doAlldEdXDiscriminators) )
process.p1 = cms.Path(process.RawToDigi * process.reconstruction * process.doAlldEdXEstimators  )
process.outpath = cms.EndPath(process.RE_RECO)
process.GlobalTag.globaltag = "IDEAL_V5::All"

