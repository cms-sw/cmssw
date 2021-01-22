import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalMustacheSCParametersMakerwrite")

# Load CondDB service
process.load("CondCore.CondDB.CondDB_cfi")

# output database
process.CondDB.connect = 'sqlite_file:ecalmustachescparameters.db'

# load the ESSource and ESProducer for the record
process.load("RecoEcal.EgammaCoreTools.EcalMustacheSCParametersESProducer_cff")

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDB,
    toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('EcalMustacheSCParametersRcd'),
            tag = cms.string('EcalMustacheSCParameters_average')
        )
    )
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.ecalMustacheSCParamsMaker = cms.EDAnalyzer("EcalMustacheSCParametersMaker")

process.path = cms.Path(process.ecalMustacheSCParamsMaker)

