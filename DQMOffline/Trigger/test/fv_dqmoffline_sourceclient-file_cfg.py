import FWCore.ParameterSet.Config as cms

process = cms.Process("DQM")

#
#  DQM SERVICES
#
process.load("DQMServices.Core.DQM_cfg")

#
#  DQM SOURCES
#
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")


process.load("DQMServices.Components.DQMEnvironment_cfi")

#### Ading VR 2010/03/24
# import of standard configurations
#process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
#process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
#process.load('DQMOffline.Configuration.DQMOffline_cff')
#process.load('Configuration.StandardSequences.EndOfProcess_cff')
#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#process.load('Configuration.EventContent.EventContent_cff')

# import of standard configurations
#process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
#process.load('Configuration.StandardSequences.Harvesting_cff')




process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("curRun_files_cfi")



process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True))

process.load("DQMServices.Components.DQMStoreStats_cfi")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 1



###############################
# Only hltResults
#
##############################
#
# Offline
process.pHLT = cms.Path(process.hltResults)



process.p = cms.EndPath(process.dqmSaver)
process.DQMStore.verbose = 0
process.DQM.collectorHost = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/StreamExpress/Commissioning10-v6/FV'
