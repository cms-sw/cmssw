import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


process.source = cms.Source("EmptySource")
process.source.firstRun = cms.untracked.uint32(67838)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.connect = cms.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_31X_GLOBALTAG')

process.GlobalTag.globaltag = 'GR09_31X_V1H::All'


process.load("SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff")



process.checkTPGsModule = cms.EDAnalyzer("HcalLutGenerator",
                                         tag = cms.string('CRAFTPhysicsV2'),
                                         HO_master_file = cms.string('inputLUTcoder_CRUZET_part4_HO.dat')
                                         )


process.p = cms.Path(process.checkTPGsModule)

