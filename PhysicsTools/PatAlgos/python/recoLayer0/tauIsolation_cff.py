import FWCore.ParameterSet.Config as cms
import copy

# compute IsoDeposits from all PFCandidates
tauIsoDepositPFCandidates = cms.EDProducer("CandIsoDepositProducer",
    src = cms.InputTag("hpsPFTauProducer"),
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

        # max. distance in z-direction between tau production vertex and PFCandidates included in IsoDeposit computation
        # (cut per default disabled, since well-defined for PFCandidates associated to tracks (PFChargedHadrons) only)
        Diff_z = cms.double(1.e+4),
        # max. distance in x-y between tau production vertex and PFCandidates included in IsoDeposit computation
        # (cut per default disabled, since well-defined for PFCandidates associated to tracks (PFChargedHadrons) only)
        Diff_r = cms.double(1.e+4),

        # collection of PFTaus, needed for excluding particles in tau signal cone from IsoDeposit
        tauSource = cms.InputTag("hpsPFTauProducer"),
        # maximum distance in eta-phi, needed to match PFTau to direction passed as function argument to Extractor
        dRmatchPFTau = cms.double(0.1),
        # size of cones around tau signal cone particles excluded from IsoDeposit computation
        dRvetoPFTauSignalConeConstituents = cms.double(0.01),

        DepositLabel = cms.untracked.string('')
    )
)

# compute IsoDeposits from PFChargedHadrons
# (enable cut on z and x-y distance between tau and PFCandidate production vertex)
tauIsoDepositPFChargedHadrons = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFChargedHadrons.ExtractorPSet.candidateSource = cms.InputTag("pfAllChargedHadronsPFBRECO")
tauIsoDepositPFChargedHadrons.ExtractorPSet.Diff_z = cms.double(0.2)
tauIsoDepositPFChargedHadrons.ExtractorPSet.Diff_r = cms.double(0.1)

# compute IsoDeposits from PFNeutralHadrons
tauIsoDepositPFNeutralHadrons = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFNeutralHadrons.ExtractorPSet.candidateSource = cms.InputTag("pfAllNeutralHadronsPFBRECO")

# compute IsoDeposits from PFGammas
tauIsoDepositPFGammas = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFGammas.ExtractorPSet.candidateSource = cms.InputTag("pfAllPhotonsPFBRECO")

patPFTauIsolationTask = cms.Task(
    tauIsoDepositPFCandidates,
    tauIsoDepositPFChargedHadrons,
    tauIsoDepositPFNeutralHadrons,
    tauIsoDepositPFGammas
)
patPFTauIsolation = cms.Sequence(patPFTauIsolationTask)
