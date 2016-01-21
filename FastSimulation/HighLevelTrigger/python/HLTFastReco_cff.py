import FWCore.ParameterSet.Config as cms

#Specific reconstruction sequences for FastSimulation.
from FastSimulation.HighLevelTrigger.HLTFastRecoForJetMET_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForMuon_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForEgamma_cff import *
from FastSimulation.Tracking.hltSeeds_cff import *
from FastSimulation.Tracking.hltPixelTracks_cff import *

# The hltbegin sequence (with L1 emulator)
HLTBeginSequence = cms.Sequence(
    cms.SequencePlaceholder("offlineBeamSpot")
    +hltSeedSequence
    )

HLTBeginSequenceBPTX = cms.Sequence(HLTBeginSequence)
