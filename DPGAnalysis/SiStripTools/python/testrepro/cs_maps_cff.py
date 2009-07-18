import FWCore.ParameterSet.Config as cms

#-------------------------------------------------------------
# Digi occupancy map and unpacker bad module map
#------------------------------------------------------------
from myTKAnalyses.MapCreators.newdigismap_cfi import *
from myTKAnalyses.DigiInvestigator.unpackerbadlistmap_cfi import *

nooutevent = cms.EDFilter("AlwaysFalse")

mapout = cms.OutputModule("PoolOutputModule",
                                  fileName = cms.untracked.string(""),
                                  outputCommands = cms.untracked.vstring(
    "drop *",
    "keep *_newDigisMap_*_*",
    "keep *_unpackerBadListMap_*_*"
    ),
                                  SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring("p"))
                                  )

