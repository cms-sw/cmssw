import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronPFIsolationDeposits_cff import *

elEDIsoDepositCharged=elPFIsoDepositCharged.clone(
    src="ecalDrivenGsfElectrons"
)
elEDIsoDepositChargedAll=elPFIsoDepositChargedAll.clone(
    src="ecalDrivenGsfElectrons"
)
elEDIsoDepositGamma=elPFIsoDepositGamma.clone(
    src="ecalDrivenGsfElectrons"
)
elEDIsoDepositNeutral=elPFIsoDepositNeutral.clone(
    src="ecalDrivenGsfElectrons"
)
elEDIsoDepositPU=elPFIsoDepositPU.clone(
    src="ecalDrivenGsfElectrons"
)
electronEDIsolationDepositsTask = cms.Task(
    elEDIsoDepositCharged,
    elEDIsoDepositChargedAll,
    elEDIsoDepositGamma,
    elEDIsoDepositNeutral,
    elEDIsoDepositPU
    )
electronEDIsolationDepositsSequence = cms.Sequence(electronEDIsolationDepositsTask)
