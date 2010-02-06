import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

trackRefsForJets = cms.EDProducer("ChargedRefCandidateProducer",
    src          = cms.InputTag('trackWithVertexRefSelector'),
    particleType = cms.string('pi+')
)
