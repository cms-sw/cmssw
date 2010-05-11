import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO3")



process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration/StandardSequences/RawToDigi_Data_cff')
process.load('Configuration/StandardSequences/L1Reco_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('DQMOffline/Configuration/DQMOffline_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.GlobalTag.globaltag = cms.string('GR10_P_V4::All')

process.load("RecoJets.Configuration.RecoJPTJets_cff")
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

#### Choose techical bits 40 or 41 and coincidence with BPTX (0)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND (NOT 36 AND NOT 37 AND NOT 38 AND NOT 39)')
####
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
### 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
       '/store/data/Commissioning10/MinimumBias/RECO/v7/000/132/453/D46703F7-923C-DF11-855D-0030487C778E.root' 
)
)

process.myjetplustrack = cms.EDFilter("JetCollisionFilter",
     src1 = cms.InputTag("ak5JPTJetsL2L3"))
# Additional output definition
process.ALCARECOStreamHcalCalMinBias = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalMinBias')
    ),
    outputCommands = cms.untracked.vstring('keep *'), 
    fileName = cms.untracked.string('RECOHcalCalMinBias.root'),
    dataset = cms.untracked.PSet(
#        filterName = cms.untracked.string('StreamALCARECOHcalCalMinBias'),
        dataTier = cms.untracked.string('RECO')
    )
)


process.pathALCARECOHcalCalMinBias = cms.Path(process.hltLevel1GTSeed*process.recoJPTJets*process.ak5JPTJetsL2L3*process.myjetplustrack)
process.endjob_step = cms.Path(process.endOfProcess)
process.ALCARECOStreamHcalCalMinBiasOutPath = cms.EndPath(process.ALCARECOStreamHcalCalMinBias)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalMinBias,process.endjob_step,process.ALCARECOStreamHcalCalMinBiasOutPath)
