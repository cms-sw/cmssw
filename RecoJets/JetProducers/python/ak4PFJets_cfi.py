import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

ak4PFJets = cms.EDProducer(
    "FastjetJetProducer",
    PFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.4)
    )


ak4PFJetsCHS = ak4PFJets.clone(
    src = "pfNoPileUpJME"
)

ak4PFJetsPuppi = ak4PFJets.clone(
    src = "particleFlow",
    applyWeight = True,
    srcWeights = cms.InputTag("puppi")
)

ak4PFJetsSK = ak4PFJets.clone(
    src = "softKiller",
    useExplicitGhosts = cms.bool(True)
)

ak4PFJetsCS = ak4PFJets.clone(
    useConstituentSubtraction = cms.bool(True),
    csRParam = cms.double(0.4),
    csRho_EtaMax = ak4PFJets.Rho_EtaMax,   # Just use the same eta for both C.S. and rho by default
    useExplicitGhosts = cms.bool(True),
    doAreaFastjet = True,
    jetPtMin = 100.0
)
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(ak4PFJets, src = "pfEmptyCollection")
pp_on_AA.toModify(ak4PFJetsCHS, src = "pfEmptyCollection")
pp_on_AA.toModify(ak4PFJetsPuppi, src = "pfEmptyCollection") 
