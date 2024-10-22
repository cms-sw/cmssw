import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *


#Now prepare the iso deposits for displacedMuons
dispMuPFIsoDepositCharged=isoDepositReplace('displacedMuons1stStep','pfAllChargedHadrons')
dispMuPFIsoDepositChargedAll=isoDepositReplace('displacedMuons1stStep','pfAllChargedParticles')
dispMuPFIsoDepositNeutral=isoDepositReplace('displacedMuons1stStep','pfAllNeutralHadrons')
dispMuPFIsoDepositGamma=isoDepositReplace('displacedMuons1stStep','pfAllPhotons')
dispMuPFIsoDepositPU=isoDepositReplace('displacedMuons1stStep','pfPileUpAllChargedParticles')

displacedMuonPFIsolationDepositsTask = cms.Task(
    dispMuPFIsoDepositCharged,
    dispMuPFIsoDepositChargedAll,
    dispMuPFIsoDepositGamma,
    dispMuPFIsoDepositNeutral,
    dispMuPFIsoDepositPU
    )
displacedMuonPFIsolationDepositsSequence = cms.Sequence(displacedMuonPFIsolationDepositsTask)
