import FWCore.ParameterSet.Config as cms

process = cms.Process('SUSYDQM')

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentCosmics_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.load("DQM.Physics.susyDQM_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_0/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/2C6A8750-16DA-E511-A317-0025905A60DE.root',
        'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_0/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/ACC8B364-7FD9-E511-A138-0CC47A4D7638.root',
        'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_0/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/CC367942-80D9-E511-82BC-0025905A6066.root',
        'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_0/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/DA6B0768-7FD9-E511-A385-0025905A612E.root',
    )
)

from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_GRun', '')

process.MessageLogger = cms.Service("MessageLogger",
       destinations   = cms.untracked.vstring('detailedInfo','critical','cerr'),
       critical       = cms.untracked.PSet(threshold = cms.untracked.string('ERROR')),
       detailedInfo   = cms.untracked.PSet(threshold  = cms.untracked.string('INFO')),
       cerr           = cms.untracked.PSet(threshold  = cms.untracked.string('WARNING'))
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_MEtoEDMConverter_*_*'
    ),
    fileName = cms.untracked.string('susy_dqm.root'),
)

process.run_module = cms.Path(process.MEtoEDMConverter*process.dqmSaver)
process.outpath = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.susyAnalyzer, process.run_module, process.outpath)
