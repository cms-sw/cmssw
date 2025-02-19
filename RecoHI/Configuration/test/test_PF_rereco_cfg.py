import FWCore.ParameterSet.Config as cms

process = cms.Process('PFRECO')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/ReconstructionHeavyIons_cff')
process.load("RecoHI.Configuration.Reconstruction_hiPF_cff")
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContentHeavyIons_cff')

process.Timing = cms.Service("Timing")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2))

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_7_0/RelValHydjetQ_B0_2760GeV/GEN-SIM-RECO/MC_37Y_V4-v1/0026/88958F00-8F69-DF11-846A-00261894383C.root'
    )
)


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    #outputCommands = process.RECODEBUGEventContent.outputCommands,
    fileName = cms.untracked.string('hydjetMB_PFRECO.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    )
)

# Other statements
process.GlobalTag.globaltag = 'MC_37Y_V5::All'
process.rechits = cms.Sequence(process.siPixelRecHits * process.siStripMatchedRecHits)

# Path and EndPath definitions
process.trkreco_step = cms.Path(process.rechits * process.heavyIonTracking)
process.pfreco_step = cms.Path(process.HiParticleFlowReco)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.trkreco_step,process.pfreco_step,process.out_step)
