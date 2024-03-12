import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allGlobalMuonTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("globalMuons"),
    particleType = cms.string('mu+')
)


# foo bar baz
# 3wfbZimpRH74f
# x3dDEBqTs4a8c
