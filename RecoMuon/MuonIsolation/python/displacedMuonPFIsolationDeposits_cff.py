import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *


#Now prepare the iso deposits for displacedMuons
dispMuPFIsoDepositCharged=isoDepositReplace('filteredDisplacedMuons1stStep','pfAllChargedHadrons')
dispMuPFIsoDepositChargedAll=isoDepositReplace('filteredDisplacedMuons1stStep','pfAllChargedParticles')
dispMuPFIsoDepositNeutral=isoDepositReplace('filteredDisplacedMuons1stStep','pfAllNeutralHadrons')
dispMuPFIsoDepositGamma=isoDepositReplace('filteredDisplacedMuons1stStep','pfAllPhotons')
dispMuPFIsoDepositPU=isoDepositReplace('filteredDisplacedMuons1stStep','pfPileUpAllChargedParticles')

displacedMuonPFIsolationDepositsTask = cms.Task(
    dispMuPFIsoDepositCharged,
    dispMuPFIsoDepositChargedAll,
    dispMuPFIsoDepositGamma,
    dispMuPFIsoDepositNeutral,
    dispMuPFIsoDepositPU
    )
displacedMuonPFIsolationDepositsSequence = cms.Sequence(displacedMuonPFIsolationDepositsTask)
