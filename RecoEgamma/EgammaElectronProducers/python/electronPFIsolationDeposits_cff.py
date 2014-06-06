import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

# The following should be removed up to  <--- when moving to GED only
#Now prepare the iso deposits
elPFIsoDepositCharged=isoDepositReplace('pfElectronTranslator:pf','pfAllChargedHadrons')
elPFIsoDepositChargedAll=isoDepositReplace('pfElectronTranslator:pf','pfAllChargedParticles')
elPFIsoDepositNeutral=isoDepositReplace('pfElectronTranslator:pf','pfAllNeutralHadrons')
elPFIsoDepositGamma=isoDepositReplace('pfElectronTranslator:pf','pfAllPhotons')
#elPFIsoDepositNeutral=isoDepositReplace('pfElectronTranslator:pf','pfWeightedNeutralHadrons')
#elPFIsoDepositGamma=isoDepositReplace('pfElectronTranslator:pf','pfWeightedPhotons')
elPFIsoDepositPU=isoDepositReplace('pfElectronTranslator:pf','pfPileUpAllChargedParticles')

electronPFIsolationDepositsSequence = cms.Sequence(
    elPFIsoDepositCharged+
    elPFIsoDepositChargedAll+
    elPFIsoDepositGamma+
    elPFIsoDepositNeutral+
    elPFIsoDepositPU
    )
# <---- Up to here

#Now prepare the iso deposits
gedElPFIsoDepositCharged=isoDepositReplace('gedGsfElectronsTmp','pfAllChargedHadrons')
gedElPFIsoDepositChargedAll=isoDepositReplace('gedGsfElectronsTmp','pfAllChargedParticles')
gedElPFIsoDepositNeutral=isoDepositReplace('gedGsfElectronsTmp','pfAllNeutralHadrons')
gedElPFIsoDepositGamma=isoDepositReplace('gedGsfElectronsTmp','pfAllPhotons')
#gedElPFIsoDepositNeutral=isoDepositReplace('gedGsfElectronsTmp','pfWeightedNeutralHadrons')
#gedElPFIsoDepositGamma=isoDepositReplace('gedGsfElectronsTmp','pfWeightedPhotons')
gedElPFIsoDepositPU=isoDepositReplace('gedGsfElectronsTmp','pfPileUpAllChargedParticles')

gedElectronPFIsolationDepositsSequence = cms.Sequence(
    gedElPFIsoDepositCharged+
    gedElPFIsoDepositChargedAll+
    gedElPFIsoDepositGamma+
    gedElPFIsoDepositNeutral+
    gedElPFIsoDepositPU
    )
