import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for track based alignment using min. bias events
ALCARECOTkAlMinBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlMinBias = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
seqALCARECOTkAlMinBias = cms.Sequence(ALCARECOTkAlMinBiasHLT+ALCARECOTkAlMinBias)
ALCARECOTkAlMinBiasHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlMinBiasHLT.HLTPaths = ['HLTMinBiasEcal', 'HLTMinBiasHcal', 'HLTMinBiasPixel']
ALCARECOTkAlMinBias.filter = True ##do not store empty events	

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

