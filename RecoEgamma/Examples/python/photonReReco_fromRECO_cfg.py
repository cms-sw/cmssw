import FWCore.ParameterSet.Config as cms

process = cms.Process("RERECO")
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/EventContent/EventContent_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_31X::All'

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
  '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0008/E60958D5-0458-DE11-B0B7-000423D98E30.root',
  '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0008/C0188C68-4157-DE11-AE7E-001D09F250AF.root'


))



# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('photons.root'),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(''),
        filterName = cms.untracked.string('')
    )
)



process.conversionReReco = cms.Path(process.conversionSequence)
process.photonReReco = cms.Path(process.photonSequence)
process.endjob_step = cms.Path(process.endOfProcess)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.conversionReReco,process.photonReReco,process.endjob_step,process.out_step)



