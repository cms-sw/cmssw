# official example for PF2PAT


import FWCore.ParameterSet.Config as cms

process = cms.Process("PRUNEAOD")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)



process.load("CommonTools.ParticleFlow.Sources/source_ZtoEles_DBS_cfi")




# output ------------------------------------------------------------

process.load("Configuration.EventContent.EventContent_cff")
process.prunedAod = cms.OutputModule("PoolOutputModule",
                                     process.AODSIMEventContent,
                                     fileName = cms.untracked.string('prunedAod.root')
)
process.load("CommonTools.ParticleFlow.PF2PAT_EventContent_cff")
process.prunedAod.outputCommands.extend( process.prunedAODForPF2PATEventContent.outputCommands )

# full aod, for comparisons ------------------------------------------

process.aod = cms.OutputModule("PoolOutputModule",
                               process.AODSIMEventContent,
                               fileName = cms.untracked.string('aod.root')
)


process.outpath = cms.EndPath(
    process.prunedAod + 
    process.aod
    )


# other stuff

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10

# the following are necessary for taus:

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_31X_V1::All')
