# AlCaReco for track based alignment using Upsilon->mumu events in heavy ion data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_cff import *

ALCARECOTkAlUpsilonMuMuHIHLT = ALCARECOTkAlUpsilonMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlUpsilonMuMuHI'
)

ALCARECOTkAlUpsilonMuMuHIDCSFilter = ALCARECOTkAlUpsilonMuMuDCSFilter.clone()

ALCARECOTkAlUpsilonMuMuHIGoodMuons = ALCARECOTkAlUpsilonMuMuGoodMuons.clone()
#ALCARECOTkAlUpsilonMuMuHIRelCombIsoMuons = ALCARECOTkAlUpsilonMuMuRelCombIsoMuons.clone()
#ALCARECOTkAlUpsilonMuMuHIRelCombIsoMuons.src = 'ALCARECOTkAlUpsilonMuMuHIGoodMuons'
#ALCARECOTkAlUpsilonMuMuHIRelCombIsoMuons.cut = '(isolationR03().sumPt + isolationR03().emEt + isolationR03().hadEt)/pt  < 0.3'

ALCARECOTkAlUpsilonMuMuHI = ALCARECOTkAlUpsilonMuMu.clone(
     src = 'hiGeneralTracks'
)
ALCARECOTkAlUpsilonMuMuHI.GlobalSelector.muonSource = 'ALCARECOTkAlUpsilonMuMuHIGoodMuons'

#seqALCARECOTkAlUpsilonMuMuHI = cms.Sequence(ALCARECOTkAlUpsilonMuMuHIHLT+ALCARECOTkAlUpsilonMuMuHIDCSFilter+ALCARECOTkAlUpsilonMuMuHIGoodMuons+ALCARECOTkAlUpsilonMuMuHIRelCombIsoMuons+ALCARECOTkAlUpsilonMuMuHI)
seqALCARECOTkAlUpsilonMuMuHI = cms.Sequence(ALCARECOTkAlUpsilonMuMuHIHLT+ALCARECOTkAlUpsilonMuMuHIDCSFilter+ALCARECOTkAlUpsilonMuMuHIGoodMuons+ALCARECOTkAlUpsilonMuMuHI)
