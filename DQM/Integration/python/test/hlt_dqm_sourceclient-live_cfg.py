import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")
# for live online DQM in P5
process.load("DQM.Integration.test.inputsource_cfi")
# used in the old input source
#process.DQMEventStreamHttpReader.SelectHLTOutput = cms.untracked.string('hltOutputHLTDQM')

# for testing in lxplus
#process.load("DQM.Integration.test.fileinputsource_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")

process.load("DQM.Integration.test.environment_cfi")
process.DQMStore.referenceFileName = "/dqmdata/dqm/reference/hlt_reference.root"
#process.dqmSaver.dirName = '.'

process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.GlobalTrackingGeometryESProducer = cms.ESProducer( "GlobalTrackingGeometryESProducer" ) # for muon hlt dqm
#SiStrip Local Reco
process.SiStripDetInfoFileReader = cms.Service("SiStripDetInfoFileReader")
process.TkDetMap = cms.Service("TkDetMap")

#---- for P5 (online) DB access
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")
# Condition for lxplus
#process.load("DQM.Integration.test.FrontierCondition_GT_Offline_cfi") 

# added for hlt scalars
process.load("DQM.TrigXMonitor.HLTSeedL1LogicScalers_cfi")
# added for hlt scalars
process.hltSeedL1Logic.l1GtData = cms.InputTag("l1GtUnpack","","DQM")
process.hltSeedL1Logic.dqmFolder =    cms.untracked.string("HLT/HLTSeedL1LogicScalers_SM")

#process.p = cms.EndPath(process.hlts+process.hltsClient)
process.p = cms.EndPath(process.hltSeedL1Logic)


process.pp = cms.Path(process.dqmEnv+process.dqmSaver)
process.dqmEnv.subSystemFolder = 'HLT'
#process.hltResults.plotAll = True


### process customizations included here
from DQM.Integration.test.online_customizations_cfi import *
process = customise(process)
