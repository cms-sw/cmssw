import FWCore.ParameterSet.Config as cms

#
# Load subdetector specific common files
#
from FastSimulation.Tracking.PixelVerticesProducer_cff import *
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#Specific reconstruction sequences for FastSimulation.
from FastSimulation.HighLevelTrigger.HLTFastRecoForJetMET_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForEgamma_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForMuon_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForTau_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForB_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForL1FastJet_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForXchannel_cff import *
from FastSimulation.HighLevelTrigger.HLTFastRecoForSpecial_cff import *

# The hltbegin sequence (with L1 emulator)
HLTBeginSequence = cms.Sequence(
    cms.SequencePlaceholder("offlineBeamSpot")
    )

HLTBeginSequenceBPTX = cms.Sequence(HLTBeginSequence)
