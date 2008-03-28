# The following comments couldn't be translated into the new config version:

# multi-jet
# single jet prescaled
import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
btagElecInJet_HLT = copy.deepcopy(hltHighLevel)
btagElecInJet_HLT.HLTPaths = ['HLTMinBias', 'HLT1jet', 'HLT2jet', 'HLT3jet', 'HLT4jet', 'HLT1jetPE5', 'CandHLT1jetPE7']

