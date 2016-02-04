import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkDQM")
process.load("DQM.Physics.ewkMuLumiMonitorDQM_cfi")


process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")





process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/Physics/EWK/LumiMonitor')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(        
#"rfio:/castor/cern.ch/cms/store/data/Run2010B/Mu/RECO/PromptReco-v2/000/146/728/5C8D6727-3BCA-DF11-8D0E-00304879FBB2.root"
#"rfio:/castor/cern.ch/cms/store/data/Run2010B/Mu/RECO/PromptReco-v2/000/147/926/6662D99B-35D8-DF11-AD29-003048D37456.root"

"rfio:/castor/cern.ch/cms/store/data/Run2010B/Mu/RECO/PromptReco-v2/000/149/442/B0642007-47E6-DF11-ACC0-0030487CD16E.root"

#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_8_2/RelValZMM/GEN-SIM-RECO/START38_V9-v1/0019/62C86D62-BFAF-DF11-85B3-003048678A6C.root"
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_6_1/RelValZMM/GEN-SIM-RECO/START36_V7-v1/0020/566083FA-005D-DF11-B5E5-003048679030.root"
#"rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_6_1/RelValWM/GEN-SIM-RECO/START36_V7-v1/0020/5426D78D-055D-DF11-AE6B-003048678F84.root"
    )
)


process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('detailedInfo'),
    detailedInfo = cms.untracked.PSet(
            default = cms.untracked.PSet( limit = cms.untracked.int32(1) ),
            threshold = cms.untracked.string('INFO')
            #threshold = cms.untracked.string('ERROR'),
     )
)


process.p = cms.Path(
                     process.ewkMuLumiMonitorDQM* 
                     process.dqmSaver
                     )

