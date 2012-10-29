import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

#Now prepare the iso deposits
elPFIsoDepositCharged=isoDepositReplace('pfSelectedElectrons','pfAllChargedHadrons')
elPFIsoDepositChargedAll=isoDepositReplace('pfSelectedElectrons','pfAllChargedParticles')
elPFIsoDepositNeutral=isoDepositReplace('pfSelectedElectrons','pfAllNeutralHadrons')
elPFIsoDepositGamma=isoDepositReplace('pfSelectedElectrons','pfAllPhotons')
elPFIsoDepositPU=isoDepositReplace('pfSelectedElectrons','pfPileUpAllChargedParticles')

electronPFIsolationDepositsSequence = cms.Sequence(
    elPFIsoDepositCharged+
    elPFIsoDepositChargedAll+
    elPFIsoDepositGamma+
    elPFIsoDepositNeutral+
    elPFIsoDepositPU
    )
