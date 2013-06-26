import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allTrackCandidates = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("generalTracks"),
    particleType = cms.string('pi+')
)


