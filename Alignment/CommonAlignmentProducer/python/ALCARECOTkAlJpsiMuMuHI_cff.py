# AlCaReco for track based alignment using Jpsi->mumu events in heavy ion data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMu_cff import *

ALCARECOTkAlJpsiMuMuHIHLT = ALCARECOTkAlJpsiMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlJpsiMuMuHI'
)

ALCARECOTkAlJpsiMuMuHIDCSFilter = ALCARECOTkAlJpsiMuMuDCSFilter.clone()

ALCARECOTkAlJpsiMuMuHIGoodMuons = ALCARECOTkAlJpsiMuMuGoodMuons.clone()

ALCARECOTkAlJpsiMuMuHI = ALCARECOTkAlJpsiMuMu.clone(
     src = 'hiGeneralTracks'
)

ALCARECOTkAlJpsiMuMuHI.GlobalSelector.muonSource = 'ALCARECOTkAlJpsiMuMuHIGoodMuons'

seqALCARECOTkAlJpsiMuMuHI = cms.Sequence(ALCARECOTkAlJpsiMuMuHIHLT+ALCARECOTkAlJpsiMuMuHIDCSFilter+ALCARECOTkAlJpsiMuMuHIGoodMuons+ALCARECOTkAlJpsiMuMuHI)
