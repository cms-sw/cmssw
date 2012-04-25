import FWCore.ParameterSet.Config as cms

from RecoMET.METProducers.PFMET_cfi import pfMet
from JetMETCorrections.Configuration.JetCorrectionServices_cff import *
from CMGTools.External.puJetIDAlgo_cff import JetIdParams

calibratedAK5PFJetsForPFMEtMVA = cms.EDProducer('PFJetCorrectionProducer',
    src = cms.InputTag('ak5PFJets'),
    correctors = cms.vstring("ak5PFL1FastL2L3") # NOTE: use "ak5PFL1FastL2L3" for MC / "ak5PFL1FastL2L3Residual" for Data
)

pfMEtMVA = cms.EDProducer("PFMETProducerMVA",
    srcCorrJets = cms.InputTag('calibratedAK5PFJetsForPFMEtMVA'),
    srcUncorrJets = cms.InputTag('ak5PFJets'),
    srcPFCandidates = cms.InputTag('particleFlow'),
    srcVertices = cms.InputTag('offlinePrimaryVertices'),
    srcLeptons = cms.VInputTag(), # NOTE: you need to set this to collections of electrons, muons and tau-jets
                                  #       passing the lepton reconstruction & identification criteria applied in your analysis
    srcRho = cms.InputTag('kt6PFJets','rho'),
    globalThreshold = pfMet.globalThreshold,
    minCorrJetPt = cms.double(0.),
    inputFileNames = cms.PSet(
        U     = cms.FileInPath('RecoMET/Configuration/data/gbrmet_52.root'),
        DPhi  = cms.FileInPath('RecoMET/Configuration/data/gbrmetphi_52.root'),
        CovU1 = cms.FileInPath('RecoMET/Configuration/data/gbrmetu1cov_52.root'),
        CovU2 = cms.FileInPath('RecoMET/Configuration/data/gbrmetu2cov_52.root')                              
    ),
    dZcut = cms.double(0.1),
    impactParTkThreshold = cms.double(1.),
    tmvaWeights = cms.string("CMGTools/External/data/mva_JetID_v1.weights.xml"),
    tmvaMethod = cms.string("JetID"),
    version = cms.int32(-1),
    tmvaVariables = cms.vstring(
        "nvtx",
        "jetPt",
        "jetEta",
        "jetPhi",
        "dZ",
        "d0",
        "beta",
        "betaStar",
        "nCharged",
        "nNeutrals",
        "dRMean",
        "frac01",
        "frac02",
        "frac03",
        "frac04",
        "frac05",
    ),
    tmvaSpectators = cms.vstring(),
    JetIdParams = JetIdParams,
    verbosity = cms.int32(0)
)

pfMEtMVAsequence  = cms.Sequence(
    calibratedAK5PFJetsForPFMEtMVA
   * pfMEtMVA
)
