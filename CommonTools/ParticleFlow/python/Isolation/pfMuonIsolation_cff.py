import FWCore.ParameterSet.Config as cms

# iso deposits and isolation values, defined by the Muon POG

from  RecoMuon.MuonIsolation.muonPFIsolation_cff import *

# computing the isolation for the muons produced by PF2PAT, and not for reco muons
sourceMuons = 'pfSelectedMuons'

muPFIsoDepositCharged.src = sourceMuons
muPFIsoDepositChargedAll.src = sourceMuons
muPFIsoDepositNeutral.src = sourceMuons
muPFIsoDepositGamma.src = sourceMuons
muPFIsoDepositPU.src = sourceMuons

pfMuonIsolationSequence = cms.Sequence(
    muonPFIsolationSequence 
    )
