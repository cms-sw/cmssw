import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *


#Now prepare the iso deposits
muPFIsoDepositCharged=isoDepositReplace('muons1stStep','pfAllChargedHadrons')
muPFIsoDepositChargedAll=isoDepositReplace('muons1stStep','pfAllChargedParticles')
muPFIsoDepositNeutral=isoDepositReplace('muons1stStep','pfAllNeutralHadrons')
muPFIsoDepositGamma=isoDepositReplace('muons1stStep','pfAllPhotons')
muPFIsoDepositPU=isoDepositReplace('muons1stStep','pfPileUpAllChargedParticles')

muonPFIsolationDepositsTask = cms.Task(
    muPFIsoDepositCharged,
    muPFIsoDepositChargedAll,
    muPFIsoDepositGamma,
    muPFIsoDepositNeutral,
    muPFIsoDepositPU
    )
muonPFIsolationDepositsSequence = cms.Sequence(muonPFIsolationDepositsTask)
# foo bar baz
# SlHOP5P2eyh4S
# GDgzt36Ndr3QG
