# AlCaReco for track based alignment using Upsilon->mumu events in heavy ion data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_cff import *

ALCARECOTkAlUpsilonMuMuPAHLT = ALCARECOTkAlUpsilonMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlUpsilonMuMu'
)

ALCARECOTkAlUpsilonMuMuPADCSFilter = ALCARECOTkAlUpsilonMuMuDCSFilter.clone()

ALCARECOTkAlUpsilonMuMuPAGoodMuons = ALCARECOTkAlUpsilonMuMuGoodMuons.clone()
#ALCARECOTkAlUpsilonMuMuPARelCombIsoMuons = ALCARECOTkAlUpsilonMuMuRelCombIsoMuons.clone()
#ALCARECOTkAlUpsilonMuMuPARelCombIsoMuons.src = 'ALCARECOTkAlUpsilonMuMuPAGoodMuons'
#ALCARECOTkAlUpsilonMuMuPARelCombIsoMuons.cut = '(isolationR03().sumPt + isolationR03().emEt + isolationR03().hadEt)/pt  < 0.3'

ALCARECOTkAlUpsilonMuMuPA = ALCARECOTkAlUpsilonMuMu.clone(
     src = 'generalTracks'
)
ALCARECOTkAlUpsilonMuMuPA.GlobalSelector.muonSource = 'ALCARECOTkAlUpsilonMuMuPAGoodMuons'

#seqALCARECOTkAlUpsilonMuMuPA = cms.Sequence(ALCARECOTkAlUpsilonMuMuPAHLT+ALCARECOTkAlUpsilonMuMuPADCSFilter+ALCARECOTkAlUpsilonMuMuPAGoodMuons+ALCARECOTkAlUpsilonMuMuPARelCombIsoMuons+ALCARECOTkAlUpsilonMuMuPA)
seqALCARECOTkAlUpsilonMuMuPA = cms.Sequence(ALCARECOTkAlUpsilonMuMuPAHLT
                                            +ALCARECOTkAlUpsilonMuMuPADCSFilter
                                            +ALCARECOTkAlUpsilonMuMuPAGoodMuons
                                            +ALCARECOTkAlUpsilonMuMuPA
                                            )
