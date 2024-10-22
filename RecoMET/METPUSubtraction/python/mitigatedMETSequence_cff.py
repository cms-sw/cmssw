import FWCore.ParameterSet.Config as cms

from RecoMET.METPUSubtraction.objectSelection_cff import *


##================================================
## MVA MET sequence
##================================================
from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *
from JetMETCorrections.Configuration.DefaultJEC_cff import *
from RecoJets.JetProducers.PileupJetIDParams_cfi import JetIdParams

from RecoJets.JetProducers.kt4PFJets_cfi import kt4PFJets
kt6PFJets = kt4PFJets.clone(rParam = 0.6, doRhoFastjet = True )

calibratedAK4PFJetsForPFMVAMEt = cms.EDProducer('PFJetCorrectionProducer',
    src = cms.InputTag('ak4PFJets'),
    correctors = cms.vstring("ak4PFL1FastL2L3") # NOTE: use "ak5PFL1FastL2L3" for MC / "ak5PFL1FastL2L3Residual" for Data
)

pfMVAMEt = cms.EDProducer("PFMETProducerMVA",
    srcCorrJets = cms.InputTag('calibratedAK4PFJetsForPFMVAMEt'),
    srcUncorrJets = cms.InputTag('ak4PFJets'),
    srcPFCandidates = cms.InputTag('particleFlow'),
    srcVertices = cms.InputTag('offlinePrimaryVertices'),
    srcLeptons = cms.VInputTag('selectedElectrons',
                               'selectedMuons',
                               'selectedTaus',
                               'selectedPhotons',
                               'selectedJets'),
    minNumLeptons = cms.int32(0),                     
    srcRho = cms.InputTag('fixedGridRhoFastjetAll'),
    globalThreshold = cms.double(-1.),
    minCorrJetPt = cms.double(-1.),
    inputFileNames = cms.PSet(
       U     = cms.FileInPath('RecoMET/METPUSubtraction/data/gbrmet_53_June2013_type1.root'),
       DPhi  = cms.FileInPath('RecoMET/METPUSubtraction/data/gbrmetphi_53_June2013_type1.root'),
       CovU1 = cms.FileInPath('RecoMET/METPUSubtraction/data/gbru1cov_53_Dec2012.root'),
       CovU2 = cms.FileInPath('RecoMET/METPUSubtraction/data/gbru2cov_53_Dec2012.root')
    ),
    loadMVAfromDB = cms.bool(False),                             
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
    JetIdParams = JetIdParams,                      
    verbosity = cms.int32(0)
)

pfMVAMEtTask  = cms.Task(
    kt6PFJets,
    calibratedAK4PFJetsForPFMVAMEt,
    pfMVAMEt
)
pfMVAMEtSequence  = cms.Sequence(pfMVAMEtTask)

##================================================
## Pf No Pileup MET sequence
##================================================
pfNoPUMEtTask = cms.Task()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *
calibratedAK4PFJetsForPFNoPUMEt = cms.EDProducer('PFJetCorrectionProducer',
    src = cms.InputTag('ak4PFJets'),
    correctors = cms.vstring('ak4PFL1FastL2L3Residual') # NOTE: use "ak4PFL1FastL2L3" for MC / "ak4PFL1FastL2L3Residual" for Data
)
ak4PFJetTaskForPFNoPUMEt = cms.Task(calibratedAK4PFJetsForPFNoPUMEt)
pfNoPUMEtTask.add(ak4PFJetTaskForPFNoPUMEt)

from RecoJets.JetProducers.PileupJetID_cfi import *
puJetIdForPFNoPUMEt = pileupJetId.clone(
    algos = cms.VPSet(
        full_53x,
        cutbased,
        PhilV1
        ),
#    label = "fullId", #MM does not work for weird reasons, cannot be cloned properly
    produceJetIds    = True,
    runMvas          = True,
    jets             = "calibratedAK4PFJetsForPFNoPUMEt",
    applyJec         = False,
    inputIsCorrected = True,
    )
pfNoPUMEtTask.add(puJetIdForPFNoPUMEt)

from JetMETCorrections.Type1MET.pfMETCorrectionType0_cfi import *
pfNoPUMEtTask.add(type0PFMEtCorrection)
pfCandidateToVertexAssociationForPFNoPUMEt = pfCandidateToVertexAssociation.clone(
    MaxNumberOfAssociations = 1,	
    doReassociation         = False,
    FinalAssociation        = 1,			    
    nTrackWeight            = 0.
)
pfNoPUMEtTask.add(pfCandidateToVertexAssociationForPFNoPUMEt)
pfMETcorrType0ForPFNoPUMEt = pfMETcorrType0.clone(
    srcPFCandidateToVertexAssociations = 'pfCandidateToVertexAssociationForPFNoPUMEt'
)
pfNoPUMEtTask.add(pfMETcorrType0ForPFNoPUMEt)

jvfJetIdForPFNoPUMEt = cms.EDProducer("JVFJetIdProducer",
    srcJets = cms.InputTag('calibratedAK4PFJetsForPFNoPUMEt'),
    srcPFCandidates = cms.InputTag('particleFlow'),
    srcPFCandToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationForPFNoPUMEt'),
    srcHardScatterVertex = cms.InputTag('selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0'),
    minTrackPt = cms.double(1.),                                    
    dZcut = cms.double(0.2), # cm
    JVFcut = cms.double(0.75),
    neutralJetOption = cms.string("noPU")
)
pfNoPUMEtTask.add(jvfJetIdForPFNoPUMEt)

import RecoMET.METProducers.METSigParams_cfi as met_config

pfNoPUMEt = cms.EDProducer("NoPileUpPFMEtProducer",
    srcMEt = cms.InputTag('pfMet'),
    srcMEtCov = cms.InputTag(''), # NOTE: leave empty to take MET covariance matrix from reco::PFMET object //MM 08/29/14, bypass hardcoded as this variable has never been used so far
    srcMVAMEtData = cms.InputTag('pfNoPUMEtData'),                               
    srcLeptons = cms.VInputTag( 'selectedElectrons',
                                'selectedMuons',
                                'selectedTaus',
                                'selectedPhotons',
                                'selectedJets'),
# NOTE: you need to set this to collections of electrons, muons and tau-jets
#passing the lepton reconstruction & identification criteria applied in your analysis                               
    srcMVAMEtDataLeptonMatch = cms.InputTag('pfNoPUMEtData'),                       
    srcType0Correction = cms.InputTag('pfMETcorrType0ForPFNoPUMEt'),                    
    sfNoPUjets = cms.double(1.0),
    sfNoPUjetOffsetEnCorr = cms.double(0.0),                    
    sfPUjets = cms.double(1.0),
    sfNoPUunclChargedCands = cms.double(1.0),
    sfPUunclChargedCands = cms.double(1.0),
    sfUnclNeutralCands = cms.double(0.6),
    sfType0Correction = cms.double(1.0),
    sfLeptonIsoCones = cms.double(0.6),       
    resolution = met_config.METSignificance_params,
    sfMEtCovMin = cms.double(0.6),
    sfMEtCovMax = cms.double(1.0),                           
    saveInputs = cms.bool(True),
    verbosity = cms.int32(0)                               
)
pfNoPUMEtTask.add(pfNoPUMEt)
pfNoPUMEtSequence = cms.Sequence(pfNoPUMEtTask)

mitigatedMETTask = cms.Task(
selectionSequenceForMVANoPUMETTask,
pfMVAMEtTask,
pfNoPUMEtTask
)
mitigatedMETSequence = cms.Sequence(mitigatedMETTask)
