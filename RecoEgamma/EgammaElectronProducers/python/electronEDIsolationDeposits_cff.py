import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaElectronProducers.electronPFIsolationDeposits_cff import *

elEDIsoDepositCharged=elPFIsoDepositCharged.clone()
elEDIsoDepositCharged.src="ecalDrivenGsfElectrons"

elEDIsoDepositChargedAll=elPFIsoDepositChargedAll.clone()
elEDIsoDepositChargedAll.src="ecalDrivenGsfElectrons"

elEDIsoDepositGamma=elPFIsoDepositGamma.clone()
elEDIsoDepositGamma.src="ecalDrivenGsfElectrons"

elEDIsoDepositNeutral=elPFIsoDepositNeutral.clone()
elEDIsoDepositNeutral.src="ecalDrivenGsfElectrons"

elEDIsoDepositPU=elPFIsoDepositPU.clone()
elEDIsoDepositPU.src="ecalDrivenGsfElectrons"

electronEDIsolationDepositsTask = cms.Task(
    elEDIsoDepositCharged,
    elEDIsoDepositChargedAll,
    elEDIsoDepositGamma,
    elEDIsoDepositNeutral,
    elEDIsoDepositPU
    )
electronEDIsolationDepositsSequence = cms.Sequence(electronEDIsolationDepositsTask)
