# AlCaReco for track based alignment using Z->mumu events in heavy ion data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_cff import *

ALCARECOTkAlZMuMuPAHLT = ALCARECOTkAlZMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlZMuMu'
)

ALCARECOTkAlZMuMuPADCSFilter = ALCARECOTkAlZMuMuDCSFilter.clone()

ALCARECOTkAlZMuMuPAGoodMuons = ALCARECOTkAlZMuMuGoodMuons.clone()
#ALCARECOTkAlZMuMuPARelCombIsoMuons = ALCARECOTkAlZMuMuRelCombIsoMuons.clone()
#ALCARECOTkAlZMuMuPARelCombIsoMuons.src = 'ALCARECOTkAlZMuMuPAGoodMuons'

ALCARECOTkAlZMuMuPA = ALCARECOTkAlZMuMu.clone(
     src = 'generalTracks'
)
ALCARECOTkAlZMuMuPA.GlobalSelector.muonSource = 'ALCARECOTkAlZMuMuPAGoodMuons'

#seqALCARECOTkAlZMuMuPA = cms.Sequence(ALCARECOTkAlZMuMuPAHLT+ALCARECOTkAlZMuMuPADCSFilter+ALCARECOTkAlZMuMuPAGoodMuons+ALCARECOTkAlZMuMuPARelCombIsoMuons+ALCARECOTkAlZMuMuPA)
seqALCARECOTkAlZMuMuPA = cms.Sequence(ALCARECOTkAlZMuMuPAHLT
                                      +ALCARECOTkAlZMuMuPADCSFilter
                                      +ALCARECOTkAlZMuMuPAGoodMuons
                                      +ALCARECOTkAlZMuMuPA
                                      )
