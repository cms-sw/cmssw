import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based alignment using ZMuMu events
ALCARECOTkAlZMuMuHLT = copy.deepcopy(hltHighLevel)
import copy
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOTkAlZMuMu = copy.deepcopy(AlignmentTrackSelector)
seqALCARECOTkAlZMuMu = cms.Sequence(ALCARECOTkAlZMuMuHLT+ALCARECOTkAlZMuMu)
ALCARECOTkAlZMuMuHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlZMuMuHLT.HLTPaths = ['HLT2MuonZ']
ALCARECOTkAlZMuMu.applyBasicCuts = True
ALCARECOTkAlZMuMu.ptMin = 15.0 ##GeV

ALCARECOTkAlZMuMu.etaMin = -3.5
ALCARECOTkAlZMuMu.etaMax = 3.5
ALCARECOTkAlZMuMu.nHitMin = 0
ALCARECOTkAlZMuMu.GlobalSelector.applyIsolationtest = True
ALCARECOTkAlZMuMu.GlobalSelector.applyGlobalMuonFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyMassrangeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.minXMass = 70.0 ##GeV

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.maxXMass = 110.0 ##GeV

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.daughterMass = 0.105 ##GeV (Muons)

ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyChargeFilter = True
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.charge = 0
ALCARECOTkAlZMuMu.TwoBodyDecaySelector.applyAcoplanarityFilter = False

