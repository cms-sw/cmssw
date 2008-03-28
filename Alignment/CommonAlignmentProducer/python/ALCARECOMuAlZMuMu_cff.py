import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for muon based alignment using ZMuMu events
ALCARECOMuAlZMuMuHLT = copy.deepcopy(hltHighLevel)
import copy
from Alignment.CommonAlignmentProducer.AlignmentMuonSelector_cfi import *
ALCARECOMuAlZMuMu = copy.deepcopy(AlignmentMuonSelector)
seqALCARECOMuAlZMuMu = cms.Sequence(ALCARECOMuAlZMuMuHLT+ALCARECOMuAlZMuMu)
ALCARECOMuAlZMuMuHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOMuAlZMuMuHLT.HLTPaths = ['HLT1MuonIso', 'HLT1MuonNonIso']

