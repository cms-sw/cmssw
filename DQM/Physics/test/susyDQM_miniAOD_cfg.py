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

process.load("DQM.Physics.susyDQM_miniAOD_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
   #     'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_0/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/MINIAODSIM/80X_mcRun2_asymptotic_v4-v1/10000/466A5649-16DA-E511-99BC-0CC47A4C8E2A.root',
    #    'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_0/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/MINIAODSIM/80X_mcRun2_asymptotic_v4-v1/10000/8AD4CD46-16DA-E511-B6F1-0CC47A4C8E20.root',

    # CMSSW 8.0.3
        'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_3/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/MINIAODSIM/80X_mcRun2_asymptotic_2016_v3_gs71xNewGtHcalCust-v1/00000/3E9C10CB-70F6-E511-82A0-0025905A6068.root',
        'root://eoscms:///eos/cms/store/relval/CMSSW_8_0_3/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/MINIAODSIM/80X_mcRun2_asymptotic_2016_v3_gs71xNewGtHcalCust-v1/00000/58889BCB-70F6-E511-9599-0025905A6090.root',
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
    fileName = cms.untracked.string('susy_dqm_miniAOD.root'),
)

process.run_module = cms.Path(process.MEtoEDMConverter*process.dqmSaver)
process.outpath = cms.EndPath(process.out)
process.schedule = cms.Schedule(process.susyMiniAODAnalyzer, process.run_module, process.outpath)
