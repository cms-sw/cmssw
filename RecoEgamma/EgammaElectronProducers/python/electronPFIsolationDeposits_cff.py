import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

# The following should be removed up to  <--- when moving to GED only
#Now prepare the iso deposits
elPFIsoDepositCharged=isoDepositReplace('pfElectronTranslator:pf','pfAllChargedHadrons')
elPFIsoDepositChargedAll=isoDepositReplace('pfElectronTranslator:pf','pfAllChargedParticles')
elPFIsoDepositNeutral=isoDepositReplace('pfElectronTranslator:pf','pfAllNeutralHadrons')
elPFIsoDepositGamma=isoDepositReplace('pfElectronTranslator:pf','pfAllPhotons')
elPFIsoDepositPU=isoDepositReplace('pfElectronTranslator:pf','pfPileUpAllChargedParticles')

electronPFIsolationDepositsTask = cms.Task(
    elPFIsoDepositCharged,
    elPFIsoDepositChargedAll,
    elPFIsoDepositGamma,
    elPFIsoDepositNeutral,
    elPFIsoDepositPU
    )
electronPFIsolationDepositsSequence = cms.Sequence(electronPFIsolationDepositsTask)
# <---- Up to here
