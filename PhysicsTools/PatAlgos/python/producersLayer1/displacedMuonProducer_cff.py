import FWCore.ParameterSet.Config as cms

#from PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi import * # This should be turn on when doing the muonMatch for displacedMuons
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *

sourceMuons = cms.InputTag("displacedMuons")

patDisplacedMuons = patMuons.clone(

    # Input collection
    muonSource = sourceMuons,

    # embedding objects
    embedMuonBestTrack      = cms.bool(False),  ## embed in AOD externally stored muon best track from global pflow
    embedTunePMuonBestTrack = cms.bool(False),  ## embed in AOD externally stored muon best track from muon only
    forceBestTrackEmbedding = cms.bool(False), ## force embedding separately the best tracks even if they're already embedded e.g. as tracker or global tracks
    embedTrack          = cms.bool(False), ## embed in AOD externally stored tracker track
    embedCombinedMuon   = cms.bool(True),  ## embed in AOD externally stored combined muon track
    embedStandAloneMuon = cms.bool(True),  ## embed in AOD externally stored standalone muon track
    embedPickyMuon      = cms.bool(False),  ## embed in AOD externally stored TeV-refit picky muon track
    embedTpfmsMuon      = cms.bool(False),  ## embed in AOD externally stored TeV-refit TPFMS muon track
    embedDytMuon        = cms.bool(False),  ## embed in AOD externally stored TeV-refit DYT muon track
    embedPFCandidate    = cms.bool(False),  ## embed in AOD externally stored particle flow candidate

    # embedding of muon MET corrections for caloMET
    embedCaloMETMuonCorrs = cms.bool(False),
    # embedding of muon MET corrections for tcMET
    embedTcMETMuonCorrs   = cms.bool(False), # removed from RECO/AOD!

    # Read and store combined inverse beta
    addInverseBeta    = cms.bool(True),
    sourceMuonTimeExtra = cms.InputTag("displacedMuons","combined"), #Use combined info, not only csc or dt (need to check if this is 'on' for displaced)

    # mc matching (deactivated)
    addGenMatch   = cms.bool(False),
    embedGenMatch = cms.bool(False),
    genParticleMatch = "displacedMuonMatch", # deactivated

    # high level selections
    embedHighLevelSelection = cms.bool(False),
    beamLineSrc             = cms.InputTag("offlineBeamSpot"),
    pvSrc                   = cms.InputTag("offlinePrimaryVertices"),

    # ecal PF energy
    embedPfEcalEnergy = cms.bool(False),
    addPuppiIsolation = cms.bool(False),

    # Compute and store Mini-Isolation.
    # Implemention and a description of parameters can be found in:
    # PhysicsTools/PatUtils/src/PFIsolation.cc
    # only works in miniaod, so set to True in miniAOD_tools.py
    computeMiniIso = cms.bool(False),
    effectiveAreaVec = cms.vdouble(0.0566, 0.0562, 0.0363, 0.0119, 0.0064),
    pfCandsForMiniIso = cms.InputTag("packedPFCandidates"),
    miniIsoParams = cms.vdouble(0.05, 0.2, 10.0, 0.5, 0.0001, 0.01, 0.01, 0.01, 0.0),

    computePuppiCombinedIso = cms.bool(False),
    # Standard Muon Selectors and Jet-related observables
    # Depends on MiniIsolation, so only works in miniaod
    # Don't forget to set flags properly in miniAOD_tools.py                      
    computeMuonMVA = cms.bool(False),
    mvaTrainingFile      = cms.FileInPath("RecoMuon/MuonIdentification/data/mu_2017_BDTG.weights.xml"),
    lowPtmvaTrainingFile = cms.FileInPath("RecoMuon/MuonIdentification/data/mu_lowpt_BDTG.weights.xml"),
    recomputeBasicSelectors = cms.bool(False),
    mvaUseJec = cms.bool(False),
    mvaDrMax = cms.double(0.4),
    mvaJetTag = cms.InputTag("pfCombinedInclusiveSecondaryVertexV2BJetTags"),
    mvaL1Corrector = cms.InputTag("ak4PFCHSL1FastjetCorrector"),
    mvaL1L2L3ResCorrector = cms.InputTag("ak4PFCHSL1FastL2L3Corrector"),
    rho = cms.InputTag("fixedGridRhoFastjetCentralNeutral"),

    computeSoftMuonMVA = cms.bool(False),
    softMvaTrainingFile = cms.FileInPath("RecoMuon/MuonIdentification/data/TMVA-muonid-bmm4-B-25.weights.xml"),

    # MC Info
    muonSimInfo = cms.InputTag("displacedMuonSimClassifier"), # This module does not exists but producer check existence but itself

    # Trigger Info 
    addTriggerMatching = cms.bool(False),
    triggerObjects = cms.InputTag("slimmedPatTrigger"),
    triggerResults = cms.InputTag("TriggerResults","","HLT"),
    hltCollectionFilters = cms.vstring('*')

)

patDisplacedMuons.isoDeposits = cms.PSet()
patDisplacedMuons.isolationValues = cms.PSet()

makePatDisplacedMuonsTask = cms.Task(patDisplacedMuons)
makePatDisplacedMuons = cms.Sequence(makePatDisplacedMuonsTask)

