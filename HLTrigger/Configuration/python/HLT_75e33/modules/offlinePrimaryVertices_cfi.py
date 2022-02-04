import FWCore.ParameterSet.Config as cms

offlinePrimaryVertices = cms.EDProducer("RecoChargedRefCandidatePrimaryVertexSorter",
    assignment = cms.PSet(
        DzCutForChargedFromPUVtxs = cms.double(0.2),
        EtaMinUseDz = cms.double(-1),
        maxDistanceToJetAxis = cms.double(0.07),
        maxDtSigForPrimaryAssignment = cms.double(4.0),
        maxDxyForJetAxisAssigment = cms.double(0.1),
        maxDxyForNotReconstructedPrimary = cms.double(0.01),
        maxDxySigForNotReconstructedPrimary = cms.double(2),
        maxDzErrorForPrimaryAssignment = cms.double(0.05),
        maxDzForJetAxisAssigment = cms.double(0.1),
        maxDzForPrimaryAssignment = cms.double(0.1),
        maxDzSigForPrimaryAssignment = cms.double(5.0),
        maxJetDeltaR = cms.double(0.5),
        minJetPt = cms.double(25),
        NumOfPUVtxsForCharged = cms.uint32(0),
        OnlyUseFirstDz = cms.bool(False),
        preferHighRanked = cms.bool(False),
        PtMaxCharged = cms.double(-1),
        useTiming = cms.bool(False),
        useVertexFit = cms.bool(True)
    ),
    jets = cms.InputTag("ak4CaloJetsForTrk"),
    particles = cms.InputTag("trackRefsForJetsBeforeSorting"),
    produceAssociationToOriginalVertices = cms.bool(False),
    produceNoPileUpCollection = cms.bool(False),
    producePileUpCollection = cms.bool(False),
    produceSortedVertices = cms.bool(True),
    qualityForPrimary = cms.int32(3),
    sorting = cms.PSet(

    ),
    trackTimeResoTag = cms.InputTag(""),
    trackTimeTag = cms.InputTag(""),
    usePVMET = cms.bool(True),
    vertices = cms.InputTag("unsortedOfflinePrimaryVertices")
)
