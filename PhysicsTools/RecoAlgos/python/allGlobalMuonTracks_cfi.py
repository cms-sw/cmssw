import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allGlobalMuonTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("globalMuons"),
    particleType = cms.string('mu+')
)


