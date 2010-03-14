import FWCore.ParameterSet.Config as cms

PrescaleService = cms.Service("PrescaleService",
    prescaleTable = cms.VPSet(
      cms.PSet(
        pathName  = cms.string("HLTPath"),
        prescales = cms.vuint32(1)
      )
    ),
    lvl1Labels       = cms.vstring('default'),
    lvl1DefaultLabel = cms.untracked.string('default')
)
