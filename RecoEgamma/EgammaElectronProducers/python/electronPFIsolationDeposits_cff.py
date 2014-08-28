import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

from RecoParticleFlow.PFProducer.electronPFIsolationDeposits_cff import electronPFIsolationDepositsSequence,elPFIsoDepositCharged,elPFIsoDepositChargedAll,elPFIsoDepositGamma,elPFIsoDepositNeutral,elPFIsoDepositPU

#Now prepare the iso deposits
gedElPFIsoDepositCharged=isoDepositReplace('gedGsfElectronsTmp','pfAllChargedHadrons')
gedElPFIsoDepositChargedAll=isoDepositReplace('gedGsfElectronsTmp','pfAllChargedParticles')
gedElPFIsoDepositNeutral=isoDepositReplace('gedGsfElectronsTmp','pfAllNeutralHadrons')
gedElPFIsoDepositGamma=isoDepositReplace('gedGsfElectronsTmp','pfAllPhotons')
gedElPFIsoDepositPU=isoDepositReplace('gedGsfElectronsTmp','pfPileUpAllChargedParticles')

gedElectronPFIsolationDepositsSequence = cms.Sequence(
    gedElPFIsoDepositCharged+
    gedElPFIsoDepositChargedAll+
    gedElPFIsoDepositGamma+
    gedElPFIsoDepositNeutral+
    gedElPFIsoDepositPU
    )
