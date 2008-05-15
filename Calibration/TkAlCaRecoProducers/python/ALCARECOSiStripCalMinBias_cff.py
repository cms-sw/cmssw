import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
# AlCaReco for track based calibration using min. bias events
ALCARECOSiStripCalMinBiasHLT = copy.deepcopy(hltHighLevel)
import copy
from Calibration.TkAlCaRecoProducers.CalibrationTrackSelector_cfi import *
ALCARECOSiStripCalMinBias = copy.deepcopy(CalibrationTrackSelector)
seqALCARECOSiStripCalMinBias = cms.Sequence(ALCARECOSiStripCalMinBiasHLT+ALCARECOSiStripCalMinBias)
ALCARECOSiStripCalMinBiasHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOSiStripCalMinBiasHLT.HLTPaths = ['HLTMinBiasEcal', 'HLTMinBiasHcal', 'HLTMinBiasPixel']
ALCARECOSiStripCalMinBias.applyBasicCuts = True
ALCARECOSiStripCalMinBias.ptMin = 0.8 ##GeV

ALCARECOSiStripCalMinBias.etaMin = -3.5
ALCARECOSiStripCalMinBias.etaMax = 3.5
ALCARECOSiStripCalMinBias.nHitMin = 0

