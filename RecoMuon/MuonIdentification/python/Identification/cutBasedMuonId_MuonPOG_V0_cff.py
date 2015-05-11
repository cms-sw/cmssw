import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

cutBasedMuonId_MuonPOG_V0 = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("loose"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("tight"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

central_id_registry.register(cutBasedMuonId_MuonPOG_V0,
                             'f5fa5cfaaa0d4f8055abe71102f00661')

