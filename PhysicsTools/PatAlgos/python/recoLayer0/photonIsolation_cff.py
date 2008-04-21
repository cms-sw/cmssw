import FWCore.ParameterSet.Config as cms

import copy
from PhysicsTools.IsolationAlgos.test.mu.candIsoDepositCtfTk_cfi import *
egammaPhotonTkDeposits = copy.deepcopy(candIsoDepositCtfTk)
import copy
from PhysicsTools.IsolationAlgos.test.egamma.candIsoDepositEgammaTowers_cfi import *
egammaPhotonTowersDeposits = copy.deepcopy(candIsoDepositEgammaTowers)
import copy
from PhysicsTools.IsolationAlgos.test.egamma.candIsoDepositEgammaEcal_cfi import *
egammaPhotonEcalDeposits = copy.deepcopy(candIsoDepositEgammaEcal)
patAODPhotonIsolationLabels = cms.PSet(
    associations = cms.VInputTag(cms.InputTag("egammaPhotonTkDeposits"), cms.InputTag("egammaPhotonTowersDeposits"), cms.InputTag("egammaPhotonEcalDeposits"))
)
patAODPhotonIsolations = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    patAODPhotonIsolationLabels,
    collection = cms.InputTag("photons")
)

layer0PhotonIsolations = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    patAODPhotonIsolationLabels,
    commonLabel = cms.InputTag("patAODPhotonIsolations"),
    collection = cms.InputTag("allLayer0Photons"),
    backrefs = cms.InputTag("allLayer0Photons")
)

patAODPhotonIsolation = cms.Sequence(egammaPhotonTkDeposits*egammaPhotonTowersDeposits*egammaPhotonEcalDeposits*patAODPhotonIsolations)
patLayer0PhotonIsolation = cms.Sequence(layer0PhotonIsolations)
egammaPhotonTkDeposits.src = 'photons'
egammaPhotonTkDeposits.trackType = 'fake'
egammaPhotonTkDeposits.ExtractorPSet.DR_Max = 0.7
egammaPhotonTkDeposits.ExtractorPSet.DR_Veto = 0.0
egammaPhotonTkDeposits.ExtractorPSet.Diff_z = 1
egammaPhotonTowersDeposits.src = 'photons'
egammaPhotonTowersDeposits.ExtractorPSet.extRadius = 0.7
egammaPhotonEcalDeposits.src = 'photons'
egammaPhotonEcalDeposits.ExtractorPSet.extRadius = 0.5

