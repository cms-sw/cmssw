# AlCaReco for track based alignment using Z->mumu events in heavy ion (PA) data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_cff import *

ALCARECOTkAlZMuMuPAHLT = ALCARECOTkAlZMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlZMuMuPA'
)

ALCARECOTkAlZMuMuPADCSFilter = ALCARECOTkAlZMuMuDCSFilter.clone()

ALCARECOTkAlZMuMuPAGoodMuons = ALCARECOTkAlZMuMuGoodMuons.clone()

ALCARECOTkAlZMuMuPA = ALCARECOTkAlZMuMu.clone(
     src = 'generalTracks'
)
ALCARECOTkAlZMuMuPA.GlobalSelector.muonSource = 'ALCARECOTkAlZMuMuPAGoodMuons'

seqALCARECOTkAlZMuMuPA = cms.Sequence(ALCARECOTkAlZMuMuPAHLT
                                      +ALCARECOTkAlZMuMuPADCSFilter
                                      +ALCARECOTkAlZMuMuPAGoodMuons
                                      +ALCARECOTkAlZMuMuPA
                                      )
