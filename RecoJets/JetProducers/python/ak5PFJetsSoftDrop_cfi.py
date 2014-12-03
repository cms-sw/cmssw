import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.ak5PFJets_cfi import ak5PFJets

ak5PFJetsSoftDrop = ak5PFJets.clone(
    useSoftDrop = cms.bool(True),
    zcut = cms.double(0.1),
    beta = cms.double(0.0),
<<<<<<< HEAD
    useExplicitGhosts = cms.bool(True),
=======
    useExplicitGhosts = cms.bool(True)
>>>>>>> 1e04e02... add subjets to ak5 softdrop
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )

