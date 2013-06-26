import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
from RecoMuon.MuonIsolationProducers.muIsoResultInputBlocks_cfi import *
from RecoMuon.MuonIsolationProducers.muIsolatorBlocks_cfi import *
muIsoResultM2T = cms.EDProducer("MuIsoTrackResultProducer",
    IsolatorByDepositR03,
    VetoPSet = cms.PSet(
        SelectAll = cms.bool(True)
    ),
    RemoveOtherVetos = cms.bool(True),
    InputMuIsoDeposits = cms.VPSet(cms.PSet(
        UnweightedInputTkDeps
    ), 
        cms.PSet(
            UnweightedInputTowEcalDeps
        ), 
        cms.PSet(
            UnweightedInputTowHcalDeps
        ))
)



