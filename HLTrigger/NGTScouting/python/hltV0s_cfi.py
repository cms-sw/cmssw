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

# Generated K0s and Lambda matched to their daughter TrackingParticles and to the HLT tracks and V0s
# (validation flavours only: needs the TrackingParticles and tpToHltGeneralTrackAssociation).
# posTrkIdx/negTrkIdx index the hltGeneralTrack table, recoV0Idx the hltKshort/hltLambda table.
hltGenV0Table = cms.EDProducer("HLTV0TruthTableProducer",
    genParticles = cms.InputTag("genParticles"),
    trackingParticles = cms.InputTag("mix", "MergedTrackTruth"),
    simToReco = cms.InputTag("tpToHltGeneralTrackAssociation"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    kshorts = cms.InputTag("hltGeneralV0Candidates", "Kshort"),
    lambdas = cms.InputTag("hltGeneralV0Candidates", "Lambda"),
    maxDeltaR = cms.double(0.05),
    maxRelDeltaPt = cms.double(0.1),
    name = cms.string("hltGenV0"),
    doc = cms.string("generated K0s and Lambda with their daughters and HLT reconstruction status")
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(hltGenV0Table, trackingParticles = "mixData:MergedTrackTruth")
