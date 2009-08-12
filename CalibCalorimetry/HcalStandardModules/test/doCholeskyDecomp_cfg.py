import FWCore.ParameterSet.Config as cms

process = cms.Process("CholeskyDecomp")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.prod = cms.EDFilter("HcalCholeskyDecomp")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

process.es_ascii = cms.ESSource('HcalTextCalibrations',
    input = cms.VPSet(
        cms.PSet(
            object = cms.string('CovarianceMatrices'),
            file = cms.FileInPath("105755-MCwidths.txt")
        )
    ),
    appendToDataLabel = cms.string('reference')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.p = cms.Path(process.prod)
