import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
allStandAloneMuonTracks = cms.EDProducer("ChargedCandidateProducer",
    src = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    particleType = cms.string('mu+')
)


