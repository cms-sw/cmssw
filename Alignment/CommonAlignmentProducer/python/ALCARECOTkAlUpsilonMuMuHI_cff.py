# AlCaReco for track based alignment using Upsilon->mumu events in heavy ion (PbPb) data
import FWCore.ParameterSet.Config as cms

from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_cff import *

ALCARECOTkAlUpsilonMuMuHIHLT = ALCARECOTkAlUpsilonMuMuHLT.clone(
    eventSetupPathsKey = 'TkAlUpsilonMuMuHI'
)

ALCARECOTkAlUpsilonMuMuHIDCSFilter = ALCARECOTkAlUpsilonMuMuDCSFilter.clone()

ALCARECOTkAlUpsilonMuMuHIGoodMuons = ALCARECOTkAlUpsilonMuMuGoodMuons.clone()

ALCARECOTkAlUpsilonMuMuHI = ALCARECOTkAlUpsilonMuMu.clone(
     src = 'hiGeneralTracks'
)
ALCARECOTkAlUpsilonMuMuHI.GlobalSelector.muonSource = 'ALCARECOTkAlUpsilonMuMuHIGoodMuons'

seqALCARECOTkAlUpsilonMuMuHI = cms.Sequence(ALCARECOTkAlUpsilonMuMuHIHLT
                                            +ALCARECOTkAlUpsilonMuMuHIDCSFilter
                                            +ALCARECOTkAlUpsilonMuMuHIGoodMuons
                                            +ALCARECOTkAlUpsilonMuMuHI
                                            )
