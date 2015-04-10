import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

cutBasedMuonId_MuonPOG_V0_tight = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-tight"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("tight"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
#        cms.PSet( cutName = cms.string("TrackQuality"),
#                  maxNormalizedChi2 = cms.double(10.),
#                  minNMuonHits = cms.uint32(1),
#                  minNMatchedStations = cms.uint32(2),
#                  minNPixelHits = cms.uint32(1),
#                  minTrackerlayers = cms.uint32(6),
#                  needsAdditionalProducts = cms.bool(False),
#                  isIgnored = cms.bool(False) ),
#        cms.PSet( cutName = cms.string("VertexCut"),
#                  maxDz = cms.double(0.5),
#                  maxDxy = cms.double(0.2),
#                  needsAdditionalProducts = cms.bool(True),
#                  isIgnored = cms.bool(False) ),
    ),
)

cutBasedMuonId_MuonPOG_V0_loose = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-loose"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("loose"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    ),
)

central_id_registry.register(cutBasedMuonId_MuonPOG_V0_tight,
                             '221704865d26c6a39aa5117dc29d3655')

central_id_registry.register(cutBasedMuonId_MuonPOG_V0_loose,
                             '4c420aed7267228341efab0a8a5c6f7a')

