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

process.GlobalTag.globaltag = cms.string('GR09_R_34X_V2::All')

process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

process.load("JetMETCorrections.Configuration.ZSPJetCorrections332_cff")

#### Choose techical bits 40 or 41 and coincidence with BPTX (0)
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND (NOT 36 AND NOT 37 AND NOT 38 AND NOT 39)')
####
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
### For 219, file from RelVal
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/BAEF02C0-0BED-DE11-9EBA-00261894392F.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/A461CC43-03ED-DE11-8E44-00304867BFF2.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/8A29D1B9-07ED-DE11-BDFD-002618943843.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/26BC3350-03ED-DE11-9683-002618943885.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/24E9A529-14ED-DE11-99C2-00304867C034.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0002/24597050-03ED-DE11-A701-00261894389D.root',
        '/store/data/BeamCommissioning09/MinimumBias/RECO/Dec19thReReco_341_v1/0001/F8B5CB1B-01ED-DE11-8529-003048678FAE.root'
)
)


process.myjetplustrack = cms.EDFilter("JetCollisionFilter",
    src1 = cms.InputTag("ak5CaloJets"))

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


process.pathALCARECOHcalCalMinBias = cms.Path(process.hltLevel1GTSeed*process.myjetplustrack)
process.endjob_step = cms.Path(process.endOfProcess)
process.ALCARECOStreamHcalCalMinBiasOutPath = cms.EndPath(process.ALCARECOStreamHcalCalMinBias)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalMinBias,process.endjob_step,process.ALCARECOStreamHcalCalMinBiasOutPath)
