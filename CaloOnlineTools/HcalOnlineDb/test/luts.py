import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
process.GlobalTag.globaltag = 'CRUZET4_V1::All'

process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.checkTPGsModule = cms.EDAnalyzer("HcalLutGenerator",
                                         #tag = cms.string('CRUZET_part4_physics_v3')
                                         tag = cms.string('CRUZETPhysicsV4')
                                         )


process.p = cms.Path(process.checkTPGsModule)

