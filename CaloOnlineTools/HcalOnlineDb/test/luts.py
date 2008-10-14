import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource")
#process.source.firstRun = cms.untracked.uint32(64709)
process.source.firstRun = cms.untracked.uint32(1) 
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
process.GlobalTag.globaltag = 'CRUZET4_V1::All'
#process.GlobalTag.globaltag = 'CRUZET4_V6P::All'

process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")

process.checkTPGsModule = cms.EDAnalyzer("HcalLutGenerator",
                                         #tag = cms.string('CRUZET_part4_physics_v3')
                                         tag = cms.string('CRUZETPhysicsV4Test'),
                                         #tag = cms.string('CRUZETPhysicsV5'),
                                         HO_master_file = cms.string('inputLUTcoder_CRUZET_part4_HO.dat')
                                         )


process.p = cms.Path(process.checkTPGsModule)

