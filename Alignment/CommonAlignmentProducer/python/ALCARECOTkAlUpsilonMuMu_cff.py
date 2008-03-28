import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based alignment using Upsilon->MuMu events
ALCARECOTkAlUpsilonMuMuHLT = copy.deepcopy(hltHighLevel)
import copy
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOTkAlUpsilonMuMu = copy.deepcopy(AlignmentTrackSelector)
seqALCARECOTkAlUpsilonMuMu = cms.Sequence(ALCARECOTkAlUpsilonMuMuHLT+ALCARECOTkAlUpsilonMuMu)
ALCARECOTkAlUpsilonMuMuHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlUpsilonMuMuHLT.HLTPaths = ['HLT2MuonUpsilon']
ALCARECOTkAlUpsilonMuMu.applyBasicCuts = True
ALCARECOTkAlUpsilonMuMu.ptMin = 0.8 ##GeV

ALCARECOTkAlUpsilonMuMu.etaMin = -3.5
ALCARECOTkAlUpsilonMuMu.etaMax = 3.5
ALCARECOTkAlUpsilonMuMu.nHitMin = 0
ALCARECOTkAlUpsilonMuMu.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlUpsilonMuMu.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.minXMass = 9.25 ##GeV

ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.maxXMass = 9.8 ##GeV

ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)

ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False
ALCARECOTkAlUpsilonMuMu.TwoBodyDecaySelector.acoplanarDistance = 1 ##radian


