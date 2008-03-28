# The following comments couldn't be translated into the new config version:

# b-mu tag 1 jet (no ptrel cut)
# multi-jet
# single jet prescaled
import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
btagMuonInJet_HLT = copy.deepcopy(hltHighLevel)
btagMuonInJet_HLT.HLTPaths = ['HLTB1JetMu', 'HLT1jet', 'HLT2jet', 'HLT3jet', 'HLT4jet', 'HLT1jetPE5', 'CandHLT1jetPE7']

