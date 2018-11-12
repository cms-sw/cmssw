
import FWCore.ParameterSet.Config as cms

tmtFilter = cms.EDFilter(
    "TMTFilter",
    inputTag  = cms.InputTag("rawDataCollector"),
    mpList    = cms.untracked.vint32(0,1,2,3,4,5,6,7,8)
)
