import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.TrackJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

ca4TrackJets = cms.EDProducer(
    "FastjetJetProducer",
    TrackJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam       = cms.double(0.4)
    )

