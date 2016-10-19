# AlCaReco for track based alignment using Upsilon->mumu events in heavy ion (PA) data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_cff import *

ALCARECOTkAlUpsilonMuMuPAHLT = ALCARECOTkAlUpsilonMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlUpsilonMuMu'
)

ALCARECOTkAlUpsilonMuMuPADCSFilter = ALCARECOTkAlUpsilonMuMuDCSFilter.clone()

ALCARECOTkAlUpsilonMuMuPAGoodMuons = ALCARECOTkAlUpsilonMuMuGoodMuons.clone()

ALCARECOTkAlUpsilonMuMuPA = ALCARECOTkAlUpsilonMuMu.clone(
     src = 'generalTracks'
)
ALCARECOTkAlUpsilonMuMuPA.GlobalSelector.muonSource = 'ALCARECOTkAlUpsilonMuMuPAGoodMuons'

seqALCARECOTkAlUpsilonMuMuPA = cms.Sequence(ALCARECOTkAlUpsilonMuMuPAHLT
                                            +ALCARECOTkAlUpsilonMuMuPADCSFilter
                                            +ALCARECOTkAlUpsilonMuMuPAGoodMuons
                                            +ALCARECOTkAlUpsilonMuMuPA
                                            )
