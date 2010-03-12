import FWCore.ParameterSet.Config as cms

PrescaleService = cms.Service("PrescaleService",
    prescaleTable = cms.VPSet(
      cms.PSet(
        pathName  = cms.string("HLTPath"),
        prescales = cms.vuint32(1)
      )
    ),
    lvl1Labels       = cms.vstring('default'),
    lvl1DefaultLabel = cms.untracked.string('default'),
#
# The following is a special Service class parameter. It's value (if !="")
# denotes the name of a new PSet to store the Service configuration, which
# will be inserted into the ProcessPSet.
#
    saveConfigTo_    = cms.string("@prescale_table")
)
