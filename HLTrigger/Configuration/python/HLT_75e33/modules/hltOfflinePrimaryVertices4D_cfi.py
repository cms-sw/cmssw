import FWCore.ParameterSet.Config as cms

hltOfflinePrimaryVertices4D = cms.EDProducer("RecoChargedRefCandidatePrimaryVertexSorter",
    assignment = cms.PSet(
        DzCutForChargedFromPUVtxs = cms.double(0.2),
        EtaMinUseDz = cms.double(-1),
        NumOfPUVtxsForCharged = cms.uint32(0),
        OnlyUseFirstDz = cms.bool(False),
        PtMaxCharged = cms.double(-1),
        maxDistanceToJetAxis = cms.double(0.07),
        maxDtSigForPrimaryAssignment = cms.double(3),
        maxDxyForJetAxisAssigment = cms.double(0.1),
        maxDxyForNotReconstructedPrimary = cms.double(0.01),
        maxDxySigForNotReconstructedPrimary = cms.double(2),
        maxDzErrorForPrimaryAssignment = cms.double(0.05),
        maxDzForJetAxisAssigment = cms.double(0.1),
        maxDzForPrimaryAssignment = cms.double(0.1),
        maxDzSigForPrimaryAssignment = cms.double(5),
        maxJetDeltaR = cms.double(0.5),
        minJetPt = cms.double(25),
        preferHighRanked = cms.bool(False),
        useTiming = cms.bool(True),
        useVertexFit = cms.bool(True)
    ),
    jets = cms.InputTag("hltAk4CaloJetsForTrk"),
    mightGet = cms.optional.untracked.vstring,
    particles = cms.InputTag("hltTrackRefsForJetsBeforeSorting4D"),
    produceAssociationToOriginalVertices = cms.bool(False),
    produceNoPileUpCollection = cms.bool(False),
    producePileUpCollection = cms.bool(False),
    produceSortedVertices = cms.bool(True),
    qualityForPrimary = cms.int32(3),
    sorting = cms.PSet(

    ),
    trackTimeResoTag = cms.InputTag("hltTofPID4DnoPID","sigmat0safe"),
    trackTimeTag = cms.InputTag("hltTofPID4DnoPID","t0safe"),
    usePVMET = cms.bool(True),
    vertices = cms.InputTag("hltUnsortedOfflinePrimaryVertices4D")
)
