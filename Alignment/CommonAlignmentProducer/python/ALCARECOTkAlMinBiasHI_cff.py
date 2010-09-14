# AlCaReco for track based alignment using min. bias events in heavy ion data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import *

ALCARECOTkAlMinBiasHINOTHLT = ALCARECOTkAlMinBiasNOTHLT.clone(
    eventSetupPathsKey = 'TkAlMinBiasNOT'
)
ALCARECOTkAlMinBiasHIHLT = ALCARECOTkAlMinBiasHLT.clone(
    eventSetupPathsKey = 'TkAlMinBias'
)
ALCARECOTkAlMinBiasHIDCSFilter = ALCARECOTkAlMinBiasDCSFilter.clone()

ALCARECOTkAlMinBiasHI = ALCARECOTkAlMinBias.clone(
     src = 'hiSelectedTracks'
)

seqALCARECOTkAlMinBiasHI = cms.Sequence(ALCARECOTkAlMinBiasHIHLT*~ALCARECOTkAlMinBiasHINOTHLT+ALCARECOTkAlMinBiasHIDCSFilter+ALCARECOTkAlMinBiasHI)
