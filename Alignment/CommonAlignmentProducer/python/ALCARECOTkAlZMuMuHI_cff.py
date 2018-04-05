# AlCaReco for track based alignment using Z->mumu events in heavy ion (PbPb) data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_cff import *

ALCARECOTkAlZMuMuHIHLT = ALCARECOTkAlZMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlZMuMuHI'
)

ALCARECOTkAlZMuMuHIDCSFilter = ALCARECOTkAlZMuMuDCSFilter.clone()

ALCARECOTkAlZMuMuHIGoodMuons = ALCARECOTkAlZMuMuGoodMuons.clone()

ALCARECOTkAlZMuMuHI = ALCARECOTkAlZMuMu.clone(
     src = 'hiGeneralTracks'
)
ALCARECOTkAlZMuMuHI.GlobalSelector.muonSource = 'ALCARECOTkAlZMuMuHIGoodMuons'

seqALCARECOTkAlZMuMuHI = cms.Sequence(ALCARECOTkAlZMuMuHIHLT
                                      +ALCARECOTkAlZMuMuHIDCSFilter
                                      +ALCARECOTkAlZMuMuHIGoodMuons
                                      +ALCARECOTkAlZMuMuHI
                                      )
