import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

cutBasedMuonId_MuonPOG_V0_loose = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-loose"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("loose"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_medium = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-medium"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("medium"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_tight = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-tight"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("tight"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_soft = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-soft"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("soft"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_highpt = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-highpt"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("RecoMuonBaseIDCut"),
                  idName = cms.string("highpt"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

central_id_registry.register(cutBasedMuonId_MuonPOG_V0_loose.idName , '2a4de3cc886b063d40c6977639306340')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_medium.idName, 'de66b9fbcf9c88573caa37490b2283b4')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_tight.idName , '221704865d26c6a39aa5117dc29d3655')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_soft.idName  , '689863a0ce4b02b2332728e0ba25694e')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_highpt.idName, '6a6b4824cc4abadd3e02fd86999b3acb')

