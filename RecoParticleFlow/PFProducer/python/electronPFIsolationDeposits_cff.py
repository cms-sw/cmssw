import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

#Now prepare the iso deposits
elPFIsoDepositCharged=isoDepositReplace('pfSelectedElectrons','pfAllChargedHadrons')
elPFIsoDepositChargedAll=isoDepositReplace('pfSelectedElectrons','pfAllChargedParticles')
elPFIsoDepositNeutral=isoDepositReplace('pfSelectedElectrons','pfAllNeutralHadrons')
elPFIsoDepositPU=isoDepositReplace('pfSelectedElectrons','pfPileUpAllChargedParticles')
#elPFIsoDepositGamma=isoDepositReplace('pfSelectedElectrons','pfAllPhotons')
elPFIsoDepositGamma= cms.EDProducer("CandIsoDepositProducer",
                                     src = cms.InputTag("pfSelectedElectrons"),
                                     MultipleDepositsFlag = cms.bool(False),
                                     trackType = cms.string('candidate'),
                                     ExtractorPSet = cms.PSet(
                                            Diff_z = cms.double(99999.99),
                                            ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
                                            DR_Max = cms.double(1.0),
                                            Diff_r = cms.double(99999.99),
                                            inputCandView = cms.InputTag("pfAllPhotons"),
                                            DR_Veto = cms.double(0),
                                            SCMatch_Veto = cms.bool(False),
                                            MissHitSCMatch_Veto = cms.bool(True),
                                            DepositLabel = cms.untracked.string('')
                                            )
                                    )
elPFIsoDepositCharged.ExtractorPSet.DR_Veto = cms.double(0)
elPFIsoDepositChargedAll.ExtractorPSet.DR_Veto = cms.double(0)
elPFIsoDepositNeutral.ExtractorPSet.DR_Veto = cms.double(0)
elPFIsoDepositPU.ExtractorPSet.DR_Veto = cms.double(0)


electronPFIsolationDepositsSequence = cms.Sequence(
    elPFIsoDepositCharged+
    elPFIsoDepositChargedAll+
    elPFIsoDepositGamma+
    elPFIsoDepositNeutral+
    elPFIsoDepositPU
    )
