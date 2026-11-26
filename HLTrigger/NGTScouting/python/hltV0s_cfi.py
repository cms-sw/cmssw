import FWCore.ParameterSet.Config as cms

# HLT V0 candidates (from hltGeneralV0Candidates, V0Producer) stored as NanoAOD flat tables.
# The posTrkIdx/negTrkIdx columns index the hltGeneralTrack table (built from hltGeneralTracks).
hltKshortTable = cms.EDProducer("HLTV0TableProducer",
    skipNonExistingSrc = cms.bool(True),
    src = cms.InputTag("hltGeneralV0Candidates", "Kshort"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    primaryVertices = cms.InputTag("hltOfflinePrimaryVertices"),
    name = cms.string("hltKshort"),
    doc = cms.string("HLT K0s -> pi+ pi- candidates from V0Producer")
)

hltLambdaTable = hltKshortTable.clone(
    src = ("hltGeneralV0Candidates", "Lambda"),
    name = "hltLambda",
    doc = "HLT Lambda -> p pi- (and charge conjugate) candidates from V0Producer"
)
