import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")


# instantiate & configure message logger service
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO') ),
    destinations = cms.untracked.vstring('cout')
)


# define prescale table: three rows (paths), three columns (scenarios)
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
    )
)


# instantiate prescale service and configure with above defined table
process.load("FWCore.PrescaleService.PrescaleService_cfi")
process.PrescaleService.prescaleTable = prescaleTable
process.PrescaleService.lvl1Labels = cms.vstring('10E30','10E31','10E32')
process.PrescaleService.lvl1DefaultLabel = cms.string('10E31')


# empty source for testing
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
process.source = cms.Source("EmptySource")


# define modules
process.psHLT1 = cms.EDFilter("HLTPrescaler")
process.psHLT2 = cms.EDFilter("HLTPrescaler")
process.psHLT3 = cms.EDFilter("HLTPrescaler")
process.psHLT4 = cms.EDFilter("HLTPrescaler")


# define paths
process.HLT1 = cms.Path(process.psHLT1)
process.HLT2 = cms.Path(process.psHLT2)
process.HLT3 = cms.Path(process.psHLT3)
process.HLT4 = cms.Path(process.psHLT4)
