import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

caSubjetFilterGenJets = cms.EDProducer(
    "SubjetFilterJetProducer",
    GenJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam       = cms.double(1.2),
    massDropCut  = cms.double(0.667),
    asymmCut     = cms.double(0.3),
    asymmCutLater= cms.bool(False)
    )
