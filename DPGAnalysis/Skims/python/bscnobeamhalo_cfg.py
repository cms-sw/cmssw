import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIM")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# activate the following lines to get some output
#process.MessageLogger.destinations = cms.untracked.vstring("cout")
#process.MessageLogger.cout = cms.untracked.PSet(threshold = cms.untracked.string("INFO"))
process.options = cms.untracked.PSet(
 wantSummary = cms.untracked.bool(True)
)



process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

process.L1T1=process.hltLevel1GTSeed.clone()
process.L1T1.L1TechTriggerSeeding = cms.bool(True)
process.L1T1.L1SeedsLogicalExpression = cms.string('(40 OR 41) AND NOT (36 OR 37 OR 38 OR 39)')


# process.bscnobeamhalo = cms.Path(process.L1T1+~process.L1T2)
process.bscnobeamhalo = cms.Path(process.L1T1)


process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/596/7E34865F-45E2-DE11-896A-000423D98F98.root',   
'/store/data/BeamCommissioning09/MinimumBias/RECO/v2/000/123/596/7E34865F-45E2-DE11-896A-000423D98F98.root'
),   
                            secondaryFileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/596/BC0F7A9A-39E2-DE11-A501-000423D99660.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/596/D6AB55BC-3AE2-DE11-8F92-000423D999CA.root'
))

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/DPGAnalysis/Skims/python/bscnobeamhalo_cfg.py,v $'),
    annotation = cms.untracked.string('BSC skim')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)



process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('/tmp/malgeri/BSCNOBEAMHALO.root'),
    outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
    dataset = cms.untracked.PSet(
    	      dataTier = cms.untracked.string('RAW-RECO'),
    	      filterName = cms.untracked.string('BSCNOBEAMHALO')),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('bscnobeamhalo')
    )
)

process.e = cms.EndPath(process.out)

