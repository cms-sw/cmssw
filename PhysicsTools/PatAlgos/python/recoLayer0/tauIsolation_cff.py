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

#tauIsoFromDepsPFCandidates = cms.EDFilter("CandIsolatorFromDeposits",
#    deposits = cms.VPSet(
#        # compute sum of IsoDeposits within isolation cone of size 0.5 in eta-phi
#        cms.PSet(
#            src = cms.InputTag("tauIsoDepositPFCandidates"),
#            deltaR = cms.double(0.5),
#            weight = cms.string('1'),
#            vetos = cms.vstring('0.01', 'Threshold(2.0)'),
#            skipDefaultVeto = cms.bool(True),
#            mode = cms.string('sum')
#        )
#    )
#)

# compute IsoDeposits from PFChargedHadrons
tauIsoDepositPFChargedHadrons = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFChargedHadrons.ExtractorPSet.candidateSource = cms.InputTag("pfAllChargedHadrons")

#tauIsoFromDepsPFChargedHadrons = copy.deepcopy(tauIsoFromDepsPFCandidates)
#tauIsoFromDepsPFChargedHadrons.deposits[0].src = cms.InputTag("tauIsoDepositPFChargedHadrons")

# compute IsoDeposits from PFNeutralHadrons
tauIsoDepositPFNeutralHadrons = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFNeutralHadrons.ExtractorPSet.candidateSource = cms.InputTag("pfAllNeutralHadrons")

#tauIsoFromDepsPFNeutralHadrons = copy.deepcopy(tauIsoFromDepsPFCandidates)
#tauIsoFromDepsPFNeutralHadrons.deposits[0].src = cms.InputTag("tauIsoDepositPFNeutralHadrons")

# compute IsoDeposits from PFGammas
tauIsoDepositPFGammas = copy.deepcopy(tauIsoDepositPFCandidates)
tauIsoDepositPFGammas.ExtractorPSet.candidateSource = cms.InputTag("pfAllPhotons")

#tauIsoFromDepsPFGammas = copy.deepcopy(tauIsoFromDepsPFCandidates)
#tauIsoFromDepsPFGammas.deposits[0].src = cms.InputTag("tauIsoDepositPFGammas")

# define module labels for old (tk-based isodeposit) POG isolation
patAODPFTauIsolationLabels = cms.VInputTag(
    cms.InputTag("tauIsoDepositPFCandidates"),
    cms.InputTag("tauIsoDepositPFChargedHadrons"),
    cms.InputTag("tauIsoDepositPFNeutralHadrons"),
    cms.InputTag("tauIsoDepositPFGammas")
)

# read and convert to ValueMap<IsoDeposit> keyed to Candidate;
# to be run on AOD before PAT
patAODPFTauIsolation = cms.EDFilter("MultipleIsoDepositsToValueMaps",
    collection   = cms.InputTag("pfRecoTauProducer"),
    associations = patAODPFTauIsolationLabels
)

# re-key to the candidates;
# to be run at the end of PAT Layer 0
patLayer0PFTauIsolation = cms.EDFilter("CandManyValueMapsSkimmerIsoDeposits",
    collection   = cms.InputTag("allLayer0Taus"),
    backrefs     = cms.InputTag("allLayer0Taus"),
    commonLabel  = cms.InputTag("patAODPFTauIsolation"),
    associations = patAODPFTauIsolationLabels
)

patPFTauIsolation = cms.Sequence( tauIsoDepositPFCandidates #* tauIsoFromDepsPFCandidates
                                * tauIsoDepositPFChargedHadrons #* tauIsoFromDepsPFChargedHadrons
                                * tauIsoDepositPFNeutralHadrons #* tauIsoFromDepsPFNeutralHadrons
                                * tauIsoDepositPFGammas #* tauIsoFromDepsPFGammas
                                )

