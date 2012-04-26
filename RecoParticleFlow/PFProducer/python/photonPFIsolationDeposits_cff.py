import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

#Now prepare the iso deposits
phPFIsoDepositCharged=isoDepositReplace('pfSelectedPhotons','pfAllChargedHadrons')
phPFIsoDepositChargedAll=isoDepositReplace('pfSelectedPhotons','pfAllChargedParticles')
phPFIsoDepositNeutral=isoDepositReplace('pfSelectedPhotons','pfAllNeutralHadrons')
#phPFIsoDepositGamma=isoDepositReplace('pfSelectedPhotons','pfAllPhotons')
phPFIsoDepositPU=isoDepositReplace('pfSelectedPhotons','pfPileUpAllChargedParticles')
phPFIsoDepositGamma= cms.EDProducer("CandIsoDepositProducer",
                                    src = cms.InputTag("pfSelectedPhotons"),
                                    MultipleDepositsFlag = cms.bool(False),
                                    trackType = cms.string('candidate'),
                                    ExtractorPSet = cms.PSet(
                                        Diff_z = cms.double(99999.99),
                                        ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
                                        DR_Max = cms.double(1.0),
                                        Diff_r = cms.double(99999.99),
                                        inputCandView = cms.InputTag("pfAllPhotons"),
                                        DR_Veto = cms.double(1e-05),
                                        SCMatch_Veto = cms.bool(True),
                                        DepositLabel = cms.untracked.string('')
                                        )
                            )

photonPFIsolationDepositsSequence = cms.Sequence(
    phPFIsoDepositCharged+
    phPFIsoDepositChargedAll+
    phPFIsoDepositGamma+
    phPFIsoDepositNeutral+
    phPFIsoDepositPU
    )
