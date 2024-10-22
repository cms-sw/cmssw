import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *


#Now prepare the iso deposits
muPFIsoDepositChargedPFBRECO=isoDepositReplace('pfSelectedMuonsPFBRECO','pfAllChargedHadronsPFBRECO')
muPFIsoDepositChargedAllPFBRECO=isoDepositReplace('pfSelectedMuonsPFBRECO','pfAllChargedParticlesPFBRECO')
muPFIsoDepositNeutralPFBRECO=isoDepositReplace('pfSelectedMuonsPFBRECO','pfAllNeutralHadronsPFBRECO')
muPFIsoDepositGammaPFBRECO=isoDepositReplace('pfSelectedMuonsPFBRECO','pfAllPhotonsPFBRECO')
muPFIsoDepositPUPFBRECO=isoDepositReplace('pfSelectedMuonsPFBRECO','pfPileUpAllChargedParticlesPFBRECO')

muonPFIsolationDepositsPFBRECOTask = cms.Task(
    muPFIsoDepositChargedPFBRECO,
    muPFIsoDepositChargedAllPFBRECO,
    muPFIsoDepositGammaPFBRECO,
    muPFIsoDepositNeutralPFBRECO,
    muPFIsoDepositPUPFBRECO
    )
muonPFIsolationDepositsPFBRECOSequence = cms.Sequence(muonPFIsolationDepositsPFBRECOTask)
