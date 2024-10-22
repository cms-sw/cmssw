import FWCore.ParameterSet.Config as cms
from CommonTools.RecoAlgos.primaryVertexAssociation_cfi import primaryVertexAssociation
from PhysicsTools.PatAlgos.slimming.packedPFCandidates_cfi import packedPFCandidates

lostTracks = cms.EDProducer("PATLostTracks",
    inputCandidates = cms.InputTag("particleFlow"),
    packedPFCandidates	= cms.InputTag("packedPFCandidates"),
    inputTracks = cms.InputTag("generalTracks"),
    secondaryVertices = cms.InputTag("inclusiveSecondaryVertices"),
    kshorts=cms.InputTag("generalV0Candidates","Kshort"),
    lambdas=cms.InputTag("generalV0Candidates","Lambda"),
    primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    originalVertices = cms.InputTag("offlinePrimaryVertices"),
    muons = cms.InputTag("muons"),
    minPt = cms.double(0.95),
    minHits = cms.uint32(8),
    minPixelHits = cms.uint32(1),
    covarianceVersion = cms.int32(0), #so far: 0 is Phase0, 1 is Phase1
    covariancePackingSchemas = packedPFCandidates.covariancePackingSchemas,
    qualsToAutoAccept = cms.vstring("highPurity"),
    minPtToStoreProps = cms.double(0.95),
    minPtToStoreLowQualityProps = cms.double(0.0),
    passThroughCut = cms.string("pt>2"),
    pvAssignment = primaryVertexAssociation.assignment,
    useLegacySetup = cms.bool(False), #When True: check only if track used to fit vertex[0] and do not store track detailed info for Pt between 0.5 and minPtToStoreProps GeV
    xiSelection = cms.bool(True),
    xiMassCut = cms.double(1.5)
)
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(lostTracks, covarianceVersion =1 )

from Configuration.ProcessModifiers.run2_miniAOD_pp_on_AA_103X_cff import run2_miniAOD_pp_on_AA_103X
run2_miniAOD_pp_on_AA_103X.toModify(lostTracks,inputCandidates = 'cleanedParticleFlow')

from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
run3_upc.toModify(lostTracks, minPtToStoreProps = 0.0, passThroughCut = "")
