# The following comments couldn't be translated into the new config version:

# b-mu tag 1 jet (no ptrel cut)
# multi-jet
# single jet prescaled
import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
btagMuonInJet_HLT = copy.deepcopy(hltHighLevel)
btagMuonInJet_HLT.HLTPaths = ['HLT_BTagMu_Jet20_Calib', 'HLT1jet', 'HLT_DoubleJet150', 'HLT_TripleJet85', 'HLT_QuadJet60', 
    'HLT1jetPE5', 'CandHLT1jetPE7']

