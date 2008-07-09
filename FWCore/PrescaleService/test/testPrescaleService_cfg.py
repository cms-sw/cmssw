import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.PrescaleService = cms.Service("PrescaleService",
    prescaleTable = cms.VPSet(
        cms.PSet(
            pathName = cms.string('HLT2'),
            prescales = cms.vuint32(2, 5, 10)
        ), 
        cms.PSet(
            pathName = cms.string('HLT3'),
            prescales = cms.vuint32(5, 10, 20)
        ), 
        cms.PSet(
            pathName = cms.string('HLT4'),
            prescales = cms.vuint32(10, 20, 0)
        )),
    lvl1Labels = cms.vstring('10E30', 
        '10E31', 
        '10E32'),
    lvl1DefaultLabel = cms.untracked.string('10E31')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("EmptySource")


process.psHLT1 = cms.EDFilter("HLTPrescaler")
process.psHLT2 = cms.EDFilter("HLTPrescaler")
process.psHLT3 = cms.EDFilter("HLTPrescaler")
process.psHLT4 = cms.EDFilter("HLTPrescaler")

process.HLT1 = cms.Path(process.psHLT1)
process.HLT2 = cms.Path(process.psHLT2)
process.HLT3 = cms.Path(process.psHLT3)
process.HLT4 = cms.Path(process.psHLT4)
