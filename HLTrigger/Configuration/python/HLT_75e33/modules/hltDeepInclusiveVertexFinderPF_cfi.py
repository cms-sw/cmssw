import FWCore.ParameterSet.Config as cms

hltDeepInclusiveVertexFinderPF = cms.EDProducer("InclusiveCandidateVertexFinder",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    clusterizer = cms.PSet(
        clusterMaxDistance = cms.double(0.05),
        clusterMaxSignificance = cms.double(4.5),
        clusterMinAngleCosine = cms.double(0.5),
        distanceRatio = cms.double(20),
        maxTimeSignificance = cms.double(3.5),
        seedMax3DIPSignificance = cms.double(9999),
        seedMax3DIPValue = cms.double(9999),
        seedMin3DIPSignificance = cms.double(1.2),
        seedMin3DIPValue = cms.double(0.005)
    ),
    fitterRatio = cms.double(0.25),
    fitterSigmacut = cms.double(3),
    fitterTini = cms.double(256),
    maxNTracks = cms.uint32(15),
    maximumLongitudinalImpactParameter = cms.double(0.2),
    maximumTimeSignificance = cms.double(3),
    mightGet = cms.optional.untracked.vstring,
    minHits = cms.uint32(8),
    minPt = cms.double(1.4),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    tracks = cms.InputTag("particleFlowTmp"),
    useDirectVertexFitter = cms.bool(True),
    useVertexReco = cms.bool(True),
    vertexMinAngleCosine = cms.double(0.95),
    vertexMinDLen2DSig = cms.double(2.5),
    vertexMinDLenSig = cms.double(0.5),
    vertexReco = cms.PSet(
        finder = cms.string('avr'),
        primcut = cms.double(1),
        seccut = cms.double(3),
        smoothing = cms.bool(True)
    )
)
