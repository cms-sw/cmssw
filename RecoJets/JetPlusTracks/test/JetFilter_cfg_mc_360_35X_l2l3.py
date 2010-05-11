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

process.GlobalTag.globaltag = cms.string('START3X_V26A::All')

process.load("RecoJets.Configuration.RecoJPTJets_cff")
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
### 
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_5_7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START3X_V26-v1/0012/4C7F6C25-6949-DF11-830C-003048679266.root' 
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


process.pathALCARECOHcalCalMinBias = cms.Path(process.recoJPTJets*process.ak5JPTJetsL2L3*process.myjetplustrack)
process.endjob_step = cms.Path(process.endOfProcess)
process.ALCARECOStreamHcalCalMinBiasOutPath = cms.EndPath(process.ALCARECOStreamHcalCalMinBias)

# Schedule definition
process.schedule = cms.Schedule(process.pathALCARECOHcalCalMinBias,process.endjob_step,process.ALCARECOStreamHcalCalMinBiasOutPath)
