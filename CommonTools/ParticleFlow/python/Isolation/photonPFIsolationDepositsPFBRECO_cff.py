import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

#Now prepare the iso deposits
phPFIsoDepositChargedPFBRECO=isoDepositReplace('pfSelectedPhotonsPFBRECO','pfAllChargedHadronsPFBRECO')
phPFIsoDepositChargedAllPFBRECO=isoDepositReplace('pfSelectedPhotonsPFBRECO','pfAllChargedParticlesPFBRECO')
phPFIsoDepositNeutralPFBRECO=isoDepositReplace('pfSelectedPhotonsPFBRECO','pfAllNeutralHadronsPFBRECO')
#phPFIsoDepositGammaPFBRECO=isoDepositReplace('pfSelectedPhotonsPFBRECO','pfAllPhotonsPFBRECO')
phPFIsoDepositPUPFBRECO=isoDepositReplace('pfSelectedPhotonsPFBRECO','pfPileUpAllChargedParticlesPFBRECO')
phPFIsoDepositGammaPFBRECO= cms.EDProducer("CandIsoDepositProducer",
                                    src = cms.InputTag("pfSelectedPhotonsPFBRECO"),
                                    MultipleDepositsFlag = cms.bool(False),
                                    trackType = cms.string('candidate'),
                                    ExtractorPSet = cms.PSet(
                                        Diff_z = cms.double(99999.99),
                                        ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
                                        DR_Max = cms.double(1.0),
                                        Diff_r = cms.double(99999.99),
                                        inputCandView = cms.InputTag("pfAllPhotonsPFBRECO"),
                                        DR_Veto = cms.double(0),
                                        SCMatch_Veto = cms.bool(True),
                                        MissHitSCMatch_Veto = cms.bool(False),
                                        DepositLabel = cms.untracked.string('')
                                        )
                            )

phPFIsoDepositChargedPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)
phPFIsoDepositChargedAllPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)
phPFIsoDepositNeutralPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)
phPFIsoDepositPUPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)

photonPFIsolationDepositsPFBRECOSequence = cms.Sequence(
    phPFIsoDepositChargedPFBRECO+
    phPFIsoDepositChargedAllPFBRECO+
    phPFIsoDepositGammaPFBRECO+
    phPFIsoDepositNeutralPFBRECO+
    phPFIsoDepositPUPFBRECO
    )
