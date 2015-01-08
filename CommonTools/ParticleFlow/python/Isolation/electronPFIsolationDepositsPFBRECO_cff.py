import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.tools_cfi import *

#Now prepare the iso deposits
elPFIsoDepositChargedPFBRECO=isoDepositReplace('pfSelectedElectronsPFBRECO','pfAllChargedHadronsPFBRECO')
elPFIsoDepositChargedAllPFBRECO=isoDepositReplace('pfSelectedElectronsPFBRECO','pfAllChargedParticlesPFBRECO')
elPFIsoDepositNeutralPFBRECO=isoDepositReplace('pfSelectedElectronsPFBRECO','pfAllNeutralHadronsPFBRECO')
elPFIsoDepositPUPFBRECO=isoDepositReplace('pfSelectedElectronsPFBRECO','pfPileUpAllChargedParticlesPFBRECO')
#elPFIsoDepositGammaPFBRECO=isoDepositReplace('pfSelectedElectronsPFBRECO','pfAllPhotonsPFBRECO')
elPFIsoDepositGammaPFBRECO= cms.EDProducer("CandIsoDepositProducer",
                                     src = cms.InputTag("pfSelectedElectronsPFBRECO"),
                                     MultipleDepositsFlag = cms.bool(False),
                                     trackType = cms.string('candidate'),
                                     ExtractorPSet = cms.PSet(
                                            Diff_z = cms.double(99999.99),
                                            ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
                                            DR_Max = cms.double(1.0),
                                            Diff_r = cms.double(99999.99),
                                            inputCandView = cms.InputTag("pfAllPhotonsPFBRECO"),
                                            DR_Veto = cms.double(0),
                                            SCMatch_Veto = cms.bool(False),
                                            MissHitSCMatch_Veto = cms.bool(True),
                                            DepositLabel = cms.untracked.string('')
                                            )
                                    )
elPFIsoDepositChargedPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)
elPFIsoDepositChargedAllPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)
elPFIsoDepositNeutralPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)
elPFIsoDepositPUPFBRECO.ExtractorPSet.DR_Veto = cms.double(0)


electronPFIsolationDepositsPFBRECOSequence = cms.Sequence(
    elPFIsoDepositChargedPFBRECO+
    elPFIsoDepositChargedAllPFBRECO+
    elPFIsoDepositGammaPFBRECO+
    elPFIsoDepositNeutralPFBRECO+
    elPFIsoDepositPUPFBRECO
    )
