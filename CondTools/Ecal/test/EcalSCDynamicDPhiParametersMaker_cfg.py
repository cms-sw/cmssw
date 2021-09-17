import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSCDynamicDPhiParametersMakerwrite")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database
process.CondDB.connect = 'sqlite_file:ecalscdynamicdphiparameters.db'

# load the ESSource and ESProducer for the record
process.load("RecoEcal.EgammaCoreTools.EcalSCDynamicDPhiParametersESProducer_cff")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalSCDynamicDPhiParametersRcd'),
            tag = cms.string('EcalSCDynamicDPhiParameters_local')
        )
    )
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.ecalSCDynamicDPhiParamsMaker = cms.EDAnalyzer("EcalSCDynamicDPhiParametersMaker")

process.path = cms.Path(process.ecalSCDynamicDPhiParamsMaker)

