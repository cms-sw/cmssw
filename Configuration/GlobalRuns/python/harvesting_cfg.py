import FWCore.ParameterSet.Config as cms

process = cms.Process("EDMtoMEConvert")
# Accumulation of globally transformed data
#
#module EDMtoMEConverter
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# execute path
#
process.load("DQMOffline.Configuration.DQMOffline_SecondStep_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
    dropMetaData = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring('file:reco2.root')
)

process.p1 = cms.Path(process.EDMtoMEConverter*process.DQMOffline_SecondStep_woHcal*process.dqmSaver)
process.maxEvents.input = -1
process.DQMStore.referenceFileName = ''
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = '/GlobalCruzet1-A/CMSSW_2_0_8-Testing/RECO'

process.GlobalTag.globaltag = 'CRZT210_DRV::All'
process.prefer("GlobalTag")


