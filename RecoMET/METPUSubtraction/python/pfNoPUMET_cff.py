import FWCore.ParameterSet.Config as cms

pfNoPUMEtTask = cms.Task()

from JetMETCorrections.Configuration.JetCorrectionServices_cff import *
calibratedAK4PFJetsForPFNoPUMEt = cms.EDProducer('PFJetCorrectionProducer',
    src = cms.InputTag('ak4PFJets'),
    correctors = cms.vstring('ak4PFL1FastL2L3') # NOTE: use "ak4PFL1FastL2L3" for MC / "ak4PFL1FastL2L3Residual" for Data
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
    srcPUSubMETDataJet = cms.InputTag('pfNoPUMEtData','jetInfos'), 
    srcPUSubMETDataPFCands = cms.InputTag('pfNoPUMEtData','pfCandInfos'),               
    srcLeptons = cms.VInputTag(), # NOTE: you need to set this to collections of electrons, muons and tau-jets
                                  #       passing the lepton reconstruction & identification criteria applied in your analysis      
    srcPUSubMETDataJetLeptonMatch = cms.InputTag('pfNoPUMEtData','jetInfos'), 
    srcPUSubMETDataPFCandsLeptonMatch = cms.InputTag('pfNoPUMEtData','pfCandInfos'),                  
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
