import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets

ak4PFJetsFiltered = ak4PFJets.clone(
    useFiltering = cms.bool(True),
    nFilt = cms.int32(3),
    rFilt = cms.double(0.3),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )


ak4PFJetsMassDropFiltered = ak4PFJetsFiltered.clone(
    useMassDropTagger = cms.bool(True),
    muCut = cms.double(0.667),
    yCut = cms.double(0.08),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets")
    )

