import FWCore.ParameterSet.Config as cms

from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *

from PhysicsTools.PatAlgos.recoLayer0.filteredDisplacedMuons_cfi import *
filteredDisplacedMuonsTask = cms.Task(filteredDisplacedMuons)


patDisplacedMuons = patMuons.clone(

    # Input collection
    muonSource = "filteredDisplacedMuons",

    # embedding objects
    embedMuonBestTrack      = True,  ## embed in AOD externally stored muon best track from global pflow
    embedTunePMuonBestTrack = True,  ## embed in AOD externally stored muon best track from muon only
    forceBestTrackEmbedding = False, ## force embedding separately the best tracks even if they're already embedded e.g. as tracker or global tracks
    embedTrack          = True, ## embed in AOD externally stored tracker track
    embedCombinedMuon   = True,  ## embed in AOD externally stored combined muon track
    embedStandAloneMuon = True,  ## embed in AOD externally stored standalone muon track
    embedPickyMuon      = False,  ## embed in AOD externally stored TeV-refit picky muon track
    embedTpfmsMuon      = False,  ## embed in AOD externally stored TeV-refit TPFMS muon track
    embedDytMuon        = False,  ## embed in AOD externally stored TeV-refit DYT muon track
    embedPFCandidate    = False,  ## embed in AOD externally stored particle flow candidate

    # embedding of muon MET corrections for caloMET
    embedCaloMETMuonCorrs = False,
    # embedding of muon MET corrections for tcMET
    embedTcMETMuonCorrs   = False, # removed from RECO/AOD!

    # Read and store combined inverse beta
    addInverseBeta    = True,
    sourceMuonTimeExtra = ["filteredDisplacedMuons","combined"], 

    # mc matching (deactivated)
    addGenMatch   = False,
    embedGenMatch = False,
    genParticleMatch = "displacedMuonMatch", # deactivated

    # high level selections
    embedHighLevelSelection = True,
    beamLineSrc             = "offlineBeamSpot",
    pvSrc                   = "offlinePrimaryVertices",

    # ecal PF energy
    embedPfEcalEnergy = False,
    addPuppiIsolation = False,

    # Compute and store Mini-Isolation.
    # Implemention and a description of parameters can be found in:
    # PhysicsTools/PatUtils/src/PFIsolation.cc
    # only works in miniaod, so set to True in miniAOD_tools.py
    computeMiniIso = False,
    effectiveAreaVec = [0.0566, 0.0562, 0.0363, 0.0119, 0.0064],
    pfCandsForMiniIso = "packedPFCandidates",
    miniIsoParams = [0.05, 0.2, 10.0, 0.5, 0.0001, 0.01, 0.01, 0.01, 0.0],

    computePuppiCombinedIso = False,
    # Standard Muon Selectors and Jet-related observables
    # Depends on MiniIsolation, so only works in miniaod
    # Don't forget to set flags properly in miniAOD_tools.py                      
    recomputeBasicSelectors = False,
    useJec = False,
    mvaDrMax = 0.4,
    mvaJetTag = "pfCombinedInclusiveSecondaryVertexV2BJetTags",
    mvaL1Corrector = "ak4PFCHSL1FastjetCorrector",
    mvaL1L2L3ResCorrector = "ak4PFCHSL1FastL2L3Corrector",
    rho = "fixedGridRhoFastjetCentralNeutral",

    computeSoftMuonMVA = False,
    softMvaTrainingFile = "RecoMuon/MuonIdentification/data/TMVA-muonid-bmm4-B-25.weights.xml",

    # MC Info
    muonSimInfo = "displacedMuonSimClassifier", # This module does not exists but producer checks existence by itself

    # Trigger Info 
    addTriggerMatching = False,
    triggerObjects = "slimmedPatTrigger",
    triggerResults = ["TriggerResults","","HLT"],
    hltCollectionFilters = ['*']

)

patDisplacedMuons.isoDeposits = cms.PSet()
patDisplacedMuons.isolationValues = cms.PSet()

# Displaced muon task filters the displacedMuons that overlap with standard muons
makePatDisplacedMuonsTask = cms.Task(
    filteredDisplacedMuonsTask,
    patDisplacedMuons
    )

makePatDisplacedMuons = cms.Sequence(makePatDisplacedMuonsTask)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(patDisplacedMuons,
                     mvaJetTag = "pfDeepCSVJetTags:probb",
)

