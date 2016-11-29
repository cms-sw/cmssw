import FWCore.ParameterSet.Config as cms

hemispheres = cms.EDFilter(
    "HLTRHemisphere",
    inputTag = cms.InputTag("ak4PFJetsCHS"),
    minJetPt = cms.double(40),
    maxEta = cms.double(3.0),
    maxNJ = cms.int32(9)
)

caloHemispheres = cms.EDFilter(
    "HLTRHemisphere",
    inputTag = cms.InputTag("ak4CaloJets"),
    minJetPt = cms.double(30),
    maxEta = cms.double(3.0),
    maxNJ = cms.int32(9)
)

hemisphereSequence = cms.Sequence(hemispheres)

caloHemisphereSequence = cms.Sequence(caloHemispheres)
