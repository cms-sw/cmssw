import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.CondDBCommon.connect = 'sqlite_file:/afs/cern.ch/user/d/dpagano/public/dati.db'
process.CondDBCommon.DBParameters.authenticationPath = './'


process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.rimon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCObImonRcd'),
        tag = cms.string('Imon_v3')
    ))
)

process.rvmon = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCObVmonRcd'),
        tag = cms.string('Vmon_v3')
    ))
)

process.rtemp = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCObTempRcd'),
        tag = cms.string('Temp_v3')
    ))
)


process.rpvss = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RPCObPVSSmapRcd'),
        tag = cms.string('PVSS_v3')
    ))
)



process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
'/store/data/Commissioning08/Cosmics/RECO/v1/000/070/195/FC7E572F-48AE-DD11-AA2A-0019DB29C5FC.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/070/195/FC5D6C50-60AE-DD11-98A3-001617C3B66C.root',
'/store/data/Commissioning08/Cosmics/RECO/v1/000/070/195/FACA249F-2FAE-DD11-968B-001617E30D4A.root'
)
)


process.demo = cms.EDAnalyzer('RiovTest')


process.p = cms.Path(process.demo)
