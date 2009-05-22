import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.CondDBCommon.connect = 'sqlite_file:dati.db'
process.CondDBCommon.DBParameters.authenticationPath = './'


process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.rn = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCObImonRcd'),
        tag = cms.string('Imon_v3')
    ))
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#'/store/data/Commissioning08/Cosmics/RECO/CRAFT_ALL_V9_225-v2/0010/BE62FAE1-44FC-DD11-8484-003048769E5F.root'
'/store/data/Commissioning08/Cosmics/RECO/v1/000/070/195/FC7E572F-48AE-DD11-AA2A-0019DB29C5FC.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/070/195/FC5D6C50-60AE-DD11-98A3-001617C3B66C.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/070/195/FACA249F-2FAE-DD11-968B-001617E30D4A.root'
)
)


process.demo = cms.EDAnalyzer('RiovTest')


process.p = cms.Path(process.demo)
