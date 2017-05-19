import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.hltTracksForMuons_cff import *
from FastSimulation.Tracking.hltElectronGsfTracks_cff import *
from FastSimulation.Tracking.hltSeeds_cff import *
from FastSimulation.Tracking.hltPixelTracks_cff import *

# The hltbegin sequence (with L1 emulator)
HLTBeginSequence = cms.Sequence(hltSeedSequence+hltPixelTracksFitter+hltPixelTracksFilter+hltPixelTracks)

HLTBeginSequenceBPTX = cms.Sequence(HLTBeginSequence)
