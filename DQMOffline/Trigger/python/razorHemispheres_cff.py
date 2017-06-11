# for Razor trigger monitoring

import FWCore.ParameterSet.Config as cms

hemispheresDQM = cms.EDFilter(
    "HLTRHemisphere",
    inputTag = cms.InputTag("ak4PFJetsCHS"),
    minJetPt = cms.double(40),
    maxEta = cms.double(3.0),
    maxNJ = cms.int32(9)
)

caloHemispheresDQM = cms.EDFilter(
    "HLTRHemisphere",
    inputTag = cms.InputTag("ak4CaloJets"),
    minJetPt = cms.double(30),
    maxEta = cms.double(3.0),
    maxNJ = cms.int32(9)
)

hemisphereDQMSequence = cms.Sequence(hemispheresDQM)

caloHemisphereDQMSequence = cms.Sequence(caloHemispheresDQM)
