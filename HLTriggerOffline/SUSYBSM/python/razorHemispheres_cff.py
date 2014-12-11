import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
from RecoJets.Configuration.RecoPFJets_cff import ak4PFJetsCHS

hemispheres = cms.EDFilter(
    "HLTRHemisphere",
    inputTag = cms.InputTag("ak4PFJetsCHS"),
    minJetPt = cms.double(40),
    maxEta = cms.double(3.0),
    maxNJ = cms.int32(9)
)

hemisphereSequence = cms.Sequence(ak4PFJets*ak4PFJetsCHS*hemispheres)
