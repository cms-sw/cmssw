import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

#Now prepare the iso deposits
phPFIsoDepositCharged=isoDepositReplace('pfSelectedPhotons','pfAllChargedHadrons')
phPFIsoDepositChargedAll=isoDepositReplace('pfSelectedPhotons','pfAllChargedParticles')
phPFIsoDepositNeutral=isoDepositReplace('pfSelectedPhotons','pfAllNeutralHadrons')
phPFIsoDepositGamma=isoDepositReplace('pfSelectedPhotons','pfAllPhotons')
phPFIsoDepositPU=isoDepositReplace('pfSelectedPhotons','pfPileUpAllChargedParticles')

photonPFIsolationDepositsSequence = cms.Sequence(
    phPFIsoDepositCharged+
    phPFIsoDepositChargedAll+
    phPFIsoDepositGamma+
    phPFIsoDepositNeutral+
    phPFIsoDepositPU
    )
