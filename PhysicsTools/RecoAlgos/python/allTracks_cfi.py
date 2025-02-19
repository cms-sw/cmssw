import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("generalTracks"),
    particleType = cms.string('pi+')
)


