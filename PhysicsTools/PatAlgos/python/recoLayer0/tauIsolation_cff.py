import FWCore.ParameterSet.Config as cms
import copy

# compute IsoDeposits from all PFCandidates
tauIsoDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("pfRecoTauProducer"),
    MultipleDepositsFlag = cms.bool(False),
    trackType = cms.string('candidate'),
    ExtractorPSet = cms.PSet(
        # PFTau specific Extractor, which allows to exclude particles within tau signal cone from IsoDeposit computation
        ComponentName = cms.string('PFTauExtractor'),
        
        # collection of PFCandidates to be used for IsoDeposit computation
        candidateSource = cms.InputTag("particleFlow"),

        # size of outer cone for which IsoDeposits are computed
        DR_Max = cms.double(1.0),
        # size of inner cone excluded from IsoDeposit computation
        DR_Veto = cms.double(0.),

        # distance in zVertex between tau production vertex and PFCandidates included in IsoDeposit computation
        Diff_z = cms.double(0.2),
        # distance in x-y between tau production vertex and PFCandidates included in IsoDeposit computation
        Diff_r = cms.double(0.1),

        # collection of PFTaus, needed for excluding particles in tau signal cone from IsoDeposit
        tauSource = cms.InputTag("pfRecoTauProducer"),
        # maximum distance in eta-phi, needed to match PFTau to direction passed as function argument to Extractor
        dRmatchPFTau = cms.double(0.1),
        # size of cones around tau signal cone particles excluded from IsoDeposit computation
        dRvetoPFTauSignalConeConstituents = cms.double(0.01),
        
        DepositLabel = cms.untracked.string('')
    )                                             
)

# compute IsoDeposits from PFChargedHadrons
tauIsoDepositPFChargedHadrons = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFChargedHadrons.ExtractorPSet.candidateSource = cms.InputTag("pfAllChargedHadrons")

# compute IsoDeposits from PFNeutralHadrons
tauIsoDepositPFNeutralHadrons = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFNeutralHadrons.ExtractorPSet.candidateSource = cms.InputTag("pfAllNeutralHadrons")

# compute IsoDeposits from PFGammas
tauIsoDepositPFGammas = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFGammas.ExtractorPSet.candidateSource = cms.InputTag("pfAllPhotons")

patPFTauIsolation = cms.Sequence( tauIsoDepositPFCandidates
                                 * tauIsoDepositPFChargedHadrons
                                 * tauIsoDepositPFNeutralHadrons
                                 * tauIsoDepositPFGammas )


