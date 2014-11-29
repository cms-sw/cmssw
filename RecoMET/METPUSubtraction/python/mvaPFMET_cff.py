import FWCore.ParameterSet.Config as cms

#from RecoMET.METProducers.PFMET_cfi import pfMet
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
from JetMETCorrections.Configuration.DefaultJEC_cff import *
##from RecoMET.METPUSubtraction.mvaPFMET_leptons_cfi import *
## CV: importing mvaPFMET_leptons_cfi breaks produceAndDiscriminateHPSPFTaus sequence
##    (hpsPFTauDiscriminationByDecayModeFinding module gets overwritten by None,
##     in case RecoTauTag/Configuration/python/RecoPFTauTag_cff.py is loaded by
##     by top-level cfg.py file before RecoMET/METPUSubtraction/python/mvaPFMET_cff.py gets loaded)
##from RecoJets.JetProducers.PileupJetIDCutParams_cfi import full_53x_wp
from RecoJets.JetProducers.PileupJetIDParams_cfi import JetIdParams

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
kt6PFJets = kt4PFJets.clone(rParam = cms.double(0.6), doRhoFastjet = True )

##from RecoMET.METPUSubtraction.mvaPFMET_db_cfi import mvaPFMEtGBRForestsFromDB

calibratedAK4PFJetsForPFMVAMEt = cms.EDProducer('PFJetCorrectionProducer',
    src = cms.InputTag('ak4PFJets'),
    correctors = cms.vstring("ak4PFL1FastL2L3") # NOTE: use "ak5PFL1FastL2L3" for MC / "ak5PFL1FastL2L3Residual" for Data
)

pfMVAMEt = cms.EDProducer("PFMETProducerMVA",
    srcCorrJets = cms.InputTag('calibratedAK4PFJetsForPFMVAMEt'),
    srcUncorrJets = cms.InputTag('ak4PFJets'),
    srcPFCandidates = cms.InputTag('particleFlow'),
    srcVertices = cms.InputTag('offlinePrimaryVertices'),
    srcLeptons = cms.VInputTag(),#"isomuons","isoelectrons","isotaus") # NOTE: you need to set this to collections of electrons, muons and tau-jets
                                 #                                             passing the lepton reconstruction & identification criteria applied in your analysis
    minNumLeptons = cms.int32(0),                     
    srcRho = cms.InputTag('kt6PFJets','rho'),
    globalThreshold = cms.double(-1.),#pfMet.globalThreshold,
    minCorrJetPt = cms.double(-1.),
    inputFileNames = cms.PSet(
       U     = cms.FileInPath('RecoMET/METPUSubtraction/data/gbrmet_53_June2013_type1.root'),
       DPhi  = cms.FileInPath('RecoMET/METPUSubtraction/data/gbrmetphi_53_June2013_type1.root'),
       CovU1 = cms.FileInPath('RecoMET/METPUSubtraction/data/gbru1cov_53_Dec2012.root'),
       CovU2 = cms.FileInPath('RecoMET/METPUSubtraction/data/gbru2cov_53_Dec2012.root')
    ),
    loadMVAfromDB = cms.bool(False),                             
   ## inputRecords = cms.PSet(
   ##     U     = cms.string('mvaPFMET_53_Dec2012_U'),
   ##     DPhi  = cms.string('mvaPFMET_53_Dec2012_DPhi'),
   ##     CovU1 = cms.string('mvaPFMET_53_Dec2012_CovU1'),
   ##     CovU2 = cms.string('mvaPFMET_53_Dec2012_CovU2')
   ## ),
   ## loadMVAfromDB = cms.bool(False),
    is42 = cms.bool(False), # CV: set this flag to true if you are running mvaPFMET in CMSSW_4_2_x
    corrector = cms.string("ak4PFL1Fastjet"),
    useType1  = cms.bool(True), 
    useOld42  = cms.bool(False),
    dZcut     = cms.double(0.1),
    impactParTkThreshold = cms.double(0.),
    tmvaWeights = cms.string("RecoJets/JetProducers/data/TMVAClassificationCategory_JetID_MET_53X_Dec2012.weights.xml.gz"),
    tmvaMethod = cms.string("JetID"),
    version = cms.int32(-1),
    cutBased = cms.bool(False),                      
    tmvaVariables = cms.vstring(
        "nvtx",
        "jetPt",
        "jetEta",
        "jetPhi",
        "dZ",
        "beta",
        "betaStar",
        "nCharged",
        "nNeutrals",
        "dR2Mean",
        "ptD",
        "frac01",
        "frac02",
        "frac03",
        "frac04",
        "frac05"
    ),
    tmvaSpectators = cms.vstring(),
    ##JetIdParams = full_53x_wp,
    JetIdParams = JetIdParams,                      
    verbosity = cms.int32(0)
)



pfMVAMEtSequence  = cms.Sequence(
    #(isomuonseq+isotauseq+isoelectronseq)*
    kt6PFJets*
    calibratedAK4PFJetsForPFMVAMEt*
    pfMVAMEt
)
