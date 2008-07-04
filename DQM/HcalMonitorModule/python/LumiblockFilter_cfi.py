# The following comments couldn't be translated into the new config version:

# lumi block must be > this value
# lumi block must be < (*not* <= ) this value
import FWCore.ParameterSet.Config as cms

lumiblockFilter = cms.EDFilter("LumiblockFilter",
    debug = cms.untracked.bool(True),
    startblock = cms.untracked.int32(0),
    endblock = cms.untracked.int32(0)
)


