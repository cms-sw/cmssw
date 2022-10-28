import FWCore.ParameterSet.Config as cms

# -*-TCL-*-
#seqs/mods to make MuIsoDeposits
from RecoMuon.MuonIsolationProducers.muIsoDeposits_cff import *
#  sequences suggested for reco (only isoDeposits are produced at this point)
muIsolation_muonsTask = cms.Task(muIsoDeposits_muonsTask)
muIsolation_muons = cms.Sequence(muIsolation_muonsTask)
muIsolation_displacedMuonsTask = cms.Task(muIsoDeposits_displacedMuonsTask)
muIsolation_displacedMuons = cms.Sequence(muIsolation_displacedMuonsTask)
muIsolation_ParamGlobalMuonsTask = cms.Task(muIsoDeposits_ParamGlobalMuonsTask)
muIsolation_ParamGlobalMuons = cms.Sequence(muIsolation_ParamGlobalMuonsTask)
muIsolation_ParamGlobalMuonsOldTask = cms.Task(muIsoDeposits_ParamGlobalMuonsOldTask)
muIsolation_ParamGlobalMuonsOld = cms.Sequence(muIsolation_ParamGlobalMuonsOldTask)
#standard sequence
muIsolationTask = cms.Task(muIsolation_muonsTask)
muIsolation = cms.Sequence(muIsolationTask)
muIsolationDisplacedTask = cms.Task(muIsolation_displacedMuonsTask)
muIsolationDisplaced = cms.Sequence(muIsolationDisplacedTask)

