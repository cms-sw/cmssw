import FWCore.ParameterSet.Config as cms

pfNoPUchsMEtSequence = cms.Sequence()

from JetMETCorrections.Type1MET.ak4PFchsJets_cff import *
pfNoPUchsMEtSequence += ak4PFchsJetsSequence

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *
calibratedAK4PFchsJetsForPFNoPUchsMEt = cms.EDProducer('PFJetCorrectionProducer',
    src = cms.InputTag('ak4PFchsJets'),
    correctors = cms.vstring('ak4PFchsL1FastL2L3Residual') # NOTE: use "ak4PFchsL1FastL2L3" for MC / "ak4PFchsL1FastL2L3Residual" for Data
)
ak4PFJetSequenceForPFNoPUchsMEt = cms.Sequence(calibratedAK4PFchsJetsForPFNoPUchsMEt)
pfNoPUchsMEtSequence += ak4PFJetSequenceForPFNoPUchsMEt

from RecoJets.JetProducers.PileupJetID_cfi import *
puJetIdForPFNoPUchsMEt = pileupJetId.clone(
    algos = cms.VPSet(
        full_53x_chs,
        cutbased
        ),
    label = cms.string("fullId"),
    produceJetIds = cms.bool(True),
    runMvas = cms.bool(True),
    jets = cms.InputTag("calibratedAK4PFchsJetsForPFNoPUchsMEt"),
    applyJec = cms.bool(False),
    inputIsCorrected = cms.bool(True),                                     
)
pfNoPUchsMEtSequence += puJetIdForPFNoPUchsMEt

from JetMETCorrections.Type1MET.pfMETCorrectionType0_cfi import *
pfNoPUchsMEtSequence += type0PFMEtCorrection
pfCandidateToVertexAssociationForPFNoPUchsMEt = pfCandidateToVertexAssociation.clone(
    MaxNumberOfAssociations = cms.int32(1),	
    doReassociation = cms.bool(False),
    FinalAssociation = cms.untracked.int32(1),			    
    nTrackWeight = cms.double(0.)
)
pfNoPUchsMEtSequence += pfCandidateToVertexAssociationForPFNoPUchsMEt
pfMETcorrType0ForPFNoPUchsMEt = pfMETcorrType0.clone(
    srcPFCandidateToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationForPFNoPUchsMEt')
)
pfNoPUchsMEtSequence += pfMETcorrType0ForPFNoPUchsMEt

##from CommonTools.RecoUtils.pfcand_assomap_cfi import PFCandAssoMap
##pfPileUpToVertexAssociation = PFCandAssoMap.clone(
##    VertexTrackAssociationMap = cms.InputTag('trackToVertexAssociation'),
##    PFCandidateCollection = cms.InputTag('pfPileUpForAK4PFchsJets')
##)
##pfNoPUchsMEtSequence += pfPileUpToVertexAssociation

jvfJetIdForPFNoPUchsMEt = cms.EDProducer("JVFJetIdProducer",
    srcJets = cms.InputTag('calibratedAK4PFchsJetsForPFNoPUchsMEt'),                                      
    srcPFCandidates = cms.InputTag('particleFlow'),
    srcPFCandToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationForPFNoPUchsMEt'),
    srcHardScatterVertex = cms.InputTag('selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0'),
    minTrackPt = cms.double(1.),                                    
    dZcut = cms.double(0.2), # cm
    JVFcut = cms.double(0.75),
    neutralJetOption = cms.string("noPU")
)
pfNoPUchsMEtSequence += jvfJetIdForPFNoPUchsMEt

import RecoMET.METProducers.METSigParams_cfi as met_config
pfNoPUchsMEtData = cms.EDProducer("PFNoPUMEtDataProducer",
    srcJets = cms.InputTag('calibratedAK4PFchsJetsForPFNoPUchsMEt'),                               
    srcJetIds = cms.InputTag('puJetIdForPFNoPUchsMEt', 'fullId'),
    #srcJetIds = cms.InputTag('jvcJetIdForPFNoPUchsMEt', 'Id'),                          
    minJetPt = cms.double(30.0), 
    jetIdSelection = cms.string('loose'),
    jetEnOffsetCorrLabel = cms.string("ak4PFchsL1Fastjet"),
    srcPFCandidates = cms.InputTag('particleFlow'),
    srcPFCandToVertexAssociations = cms.InputTag('pfCandidateToVertexAssociationForPFNoPUchsMEt'),
    ##srcPFCandidates = cms.InputTag('pfPileUpForAK4PFchsJets'),
    ##srcPFCandToVertexAssociations = cms.InputTag('pfPileUpToVertexAssociation'),
    srcJetsForMEtCov = cms.InputTag('ak4PFchsJets'),                               
    minJetPtForMEtCov = cms.double(10.), 
    srcHardScatterVertex = cms.InputTag('selectedPrimaryVertexHighestPtTrackSumForPFMEtCorrType0'),
    dZcut = cms.double(0.2), # cm
    resolution = met_config.METSignificance_params,
    verbosity = cms.int32(0)     
)
pfNoPUchsMEtSequence += pfNoPUchsMEtData

pfNoPUchsMEt = cms.EDProducer("PFNoPUMEtProducer",
    srcMEt = cms.InputTag('pfMet'),
    srcMEtCov = cms.InputTag(''), # NOTE: leave empty to take MET covariance matrix from reco::PFMET object
    srcMVAMEtData = cms.InputTag('pfNoPUchsMEtData'),
    srcLeptons = cms.VInputTag(), # NOTE: you need to set this to collections of electrons, muons and tau-jets
                                  #       passing the lepton reconstruction & identification criteria applied in your analysis
    srcMVAMEtDataLeptonMatch = cms.InputTag('pfNoPUchsMEtData'),
    srcType0Correction = cms.InputTag('pfMETcorrType0ForPFNoPUchsMEt'),                    
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
pfNoPUchsMEtSequence += pfNoPUchsMEt
