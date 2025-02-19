import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.muIsoResultInputBlocks_cfi import *
from RecoMuon.MuonIsolationProducers.muIsolatorBlocks_cfi import *
muIsoResultM2C = cms.EDProducer("MuIsoCandidateResultProducer",
    IsolatorByDepositR03,
    VetoPSet = cms.PSet(
        SelectAll = cms.bool(True)
    ),
    BeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    RemoveOtherVetos = cms.bool(True),
    InputMuIsoDeposits = cms.VPSet(cms.PSet(
        UnweightedInputTkDeps
    ), 
        cms.PSet(
            UnweightedInputTowEcalDeps
        ), 
        cms.PSet(
            UnweightedInputTowHcalDeps
        )),
    BeamlineOption = cms.string('BeamSpotFromEvent')
)



