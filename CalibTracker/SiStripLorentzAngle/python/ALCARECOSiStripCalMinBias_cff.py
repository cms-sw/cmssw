import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# AlCaReco for track based alignment using min. bias events
ALCARECOSiStripCalMinBiasHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi

ALCARECOSiStripCalMinBias = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
seqALCARECOSiStripCalMinBias = cms.Sequence(ALCARECOSiStripCalMinBiasHLT+ALCARECOSiStripCalMinBias)
ALCARECOSiStripCalMinBiasHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOSiStripCalMinBiasHLT.HLTPaths = ['HLT_MinBiasEcal', 'HLT_MinBiasHcal', 'HLT_MinBiasPixel']
ALCARECOSiStripCalMinBias.filter = True ##do not store empty events	

ALCARECOSiStripCalMinBias.applyBasicCuts = True
ALCARECOSiStripCalMinBias.ptMin = 1.5 ##GeV

ALCARECOSiStripCalMinBias.etaMin = -3.5
ALCARECOSiStripCalMinBias.etaMax = 3.5
ALCARECOSiStripCalMinBias.nHitMin = 0
ALCARECOSiStripCalMinBias.GlobalSelector.applyIsolationtest = False
ALCARECOSiStripCalMinBias.GlobalSelector.applyGlobalMuonFilter = False
ALCARECOSiStripCalMinBias.TwoBodyDecaySelector.applyMassrangeFilter = False
ALCARECOSiStripCalMinBias.TwoBodyDecaySelector.applyChargeFilter = False
ALCARECOSiStripCalMinBias.TwoBodyDecaySelector.applyAcoplanarityFilter = False

pathALCARECOSiStripCalMinBias = cms.Path(seqALCARECOSiStripCalMinBias)
