import FWCore.ParameterSet.Config as cms

from FastSimulation.HighLevelTrigger.HLTFastRecoForMuon_cff import *
from FastSimulation.Tracking.hltElectronGsfTracks_cff import *
from FastSimulation.Tracking.hltSeeds_cff import *
from FastSimulation.Tracking.hltPixelTracks_cff import *

# The hltbegin sequence (with L1 emulator)
HLTBeginSequence = cms.Sequence(hltSeedSequence+hltPixelTracks)

HLTBeginSequenceBPTX = cms.Sequence(HLTBeginSequence)
