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

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5000) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        #CMSSW_7_5_0_pre5 relvals
        #'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre5/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/26C0DAA2-180B-E511-A8AB-00261894386F.root',
        #'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre5/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/8E99B59F-180B-E511-B9EA-00248C0BE005.root',
        #'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre5/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/54D253B1-C30B-E511-8918-0025905A605E.root'
        #CMSSW_7_5_0_pre5 ttbar
        'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre5/RelValTTbarLepton_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/56FA331B-B90B-E511-93CE-0025905B858E.root',
        'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre5/RelValTTbarLepton_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/6605FE82-120B-E511-8721-00261894384F.root',
        'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre5/RelValTTbarLepton_13/GEN-SIM-RECO/MCRUN2_75_V5-v1/00000/86B3E421-B90B-E511-A5BE-003048FFCBFC.root'
        #CMSSW_7_5_0_pre4 ttbar
        #'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre4/RelValTTbarLepton_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/1050C70B-54F6-E411-8CE2-0025905A7786.root',
        #'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre4/RelValTTbarLepton_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/5811D602-54F6-E411-9821-0025905A611C.root',
        #'root://eoscms:///eos/cms/store/relval/CMSSW_7_5_0_pre4/RelValTTbarLepton_13/GEN-SIM-RECO/MCRUN2_75_V1-v1/00000/90838E13-FAF5-E411-B7DD-003048FF9AA6.root'
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
