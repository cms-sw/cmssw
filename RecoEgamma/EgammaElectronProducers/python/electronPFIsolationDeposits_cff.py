import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

#Now prepare the iso deposits
elPFIsoDepositCharged=isoDepositReplace('pfElectronTranslator:pf','pfAllChargedHadrons')
elPFIsoDepositChargedAll=isoDepositReplace('pfElectronTranslator:pf','pfAllChargedParticles')
elPFIsoDepositNeutral=isoDepositReplace('pfElectronTranslator:pf','pfAllNeutralHadrons')
elPFIsoDepositGamma=isoDepositReplace('pfElectronTranslator:pf','pfAllPhotons')
elPFIsoDepositPU=isoDepositReplace('pfElectronTranslator:pf','pfPileUpAllChargedParticles')

electronPFIsolationDepositsSequence = cms.Sequence(
    elPFIsoDepositCharged+
    elPFIsoDepositChargedAll+
    elPFIsoDepositGamma+
    elPFIsoDepositNeutral+
    elPFIsoDepositPU
    )
