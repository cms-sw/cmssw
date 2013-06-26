import FWCore.ParameterSet.Config as cms

# -*-SH-*-
# MuonCaloCompatibility
from RecoMuon.MuonIdentification.caloCompatibility_cff import *
from TrackingTools.TrackAssociator.default_cfi import *
calomuons = cms.EDProducer("CaloMuonProducer",
    # TrackDetectorAssociator
    TrackAssociatorParameterBlock,
    MuonCaloCompatibilityBlock,
    inputCollection = cms.InputTag("muons1stStep"),
    inputMuons = cms.InputTag("muons1stStep"),
    inputTracks = cms.InputTag("generalTracks"),
    minCaloCompatibility = cms.double(0.6),
    minPt = cms.double(1.0)
)



