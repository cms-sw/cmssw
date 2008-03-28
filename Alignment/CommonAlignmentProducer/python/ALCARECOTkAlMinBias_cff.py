import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based alignment using min. bias events
ALCARECOTkAlMinBiasHLT = copy.deepcopy(hltHighLevel)
import copy
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
ALCARECOTkAlMinBias = copy.deepcopy(AlignmentTrackSelector)
seqALCARECOTkAlMinBias = cms.Sequence(ALCARECOTkAlMinBiasHLT+ALCARECOTkAlMinBias)
ALCARECOTkAlMinBiasHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlMinBiasHLT.HLTPaths = ['HLTMinBias', 'HLTMinBiasPixel']
ALCARECOTkAlMinBias.applyBasicCuts = True
ALCARECOTkAlMinBias.ptMin = 1.5 ##GeV

ALCARECOTkAlMinBias.etaMin = -3.5
ALCARECOTkAlMinBias.etaMax = 3.5
ALCARECOTkAlMinBias.nHitMin = 0
ALCARECOTkAlMinBias.GlobalSelector.applyIsolationtest = False
ALCARECOTkAlMinBias.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOTkAlMinBias.TwoBodyDecaySelector.applyAcoplanarityFilter = False

