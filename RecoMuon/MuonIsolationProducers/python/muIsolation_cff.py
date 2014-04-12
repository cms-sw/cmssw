import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
#seqs/mods to make MuIsoDeposits
from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *
#  sequences suggested for reco (only isoDeposits are produced at this point)
muIsolation_muons = cms.Sequence(muIsoDeposits_muons)
muIsolation_ParamGlobalMuons = cms.Sequence(muIsoDeposits_ParamGlobalMuons)
muIsolation_ParamGlobalMuonsOld = cms.Sequence(muIsoDeposits_ParamGlobalMuonsOld)
#standard sequence
muIsolation = cms.Sequence(muIsolation_muons)


