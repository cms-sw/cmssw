import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cff import *

makeHeavyIonMuons = cms.Sequence(
    # pat and HI specifics
    muonMatch *
    # object production
    patMuons
    )


