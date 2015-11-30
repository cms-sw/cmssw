# AlCaReco for track based alignment using min. bias events in heavy ion data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import *

ALCARECOTkAlMinBiasHIHLT = ALCARECOTkAlMinBiasHLT.clone(
    eventSetupPathsKey = 'TkAlMinBiasHI'
)

ALCARECOTkAlMinBiasHIDCSFilter = ALCARECOTkAlMinBiasDCSFilter.clone()

ALCARECOTkAlMinBiasHI = ALCARECOTkAlMinBias.clone(
     src = 'hiGeneralTracks'
)

ALCARECOTkAlMinBiasHI.trackQualities = cms.vstring("highPurity")
#ALCARECOTkAlMinBiasHI.pMin = 3.0 ##GeV

seqALCARECOTkAlMinBiasHI = cms.Sequence(ALCARECOTkAlMinBiasHIHLT+ALCARECOTkAlMinBiasHIDCSFilter+ALCARECOTkAlMinBiasHI)
