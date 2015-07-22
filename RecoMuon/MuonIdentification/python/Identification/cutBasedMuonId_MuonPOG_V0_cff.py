import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

cutBasedMuonId_MuonPOG_V0_loose = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-loose"),
    isPOGApproved = cms.untracked.bool(True),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonPOGStandardCut"),
                  idName = cms.string("loose"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_medium = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-medium"),
    isPOGApproved = cms.untracked.bool(True),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonPOGStandardCut"),
                  idName = cms.string("medium"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_tight = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-tight"),
    isPOGApproved = cms.untracked.bool(True),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonPOGStandardCut"),
                  idName = cms.string("tight"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_soft = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-soft"),
    isPOGApproved = cms.untracked.bool(True),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonPOGStandardCut"),
                  idName = cms.string("soft"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

cutBasedMuonId_MuonPOG_V0_highpt = cms.PSet(
    idName = cms.string("cutBasedMuonId-MuonPOG-V0-highpt"),
    isPOGApproved = cms.untracked.bool(True),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MuonPOGStandardCut"),
                  idName = cms.string("highpt"),
                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                  needsAdditionalProducts = cms.bool(True),
                  isIgnored = cms.bool(False) ),
    )
)

central_id_registry.register(cutBasedMuonId_MuonPOG_V0_loose.idName , '2e3bc4052d5d56c437b9285185f42ed9')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_medium.idName, 'd32c5dcaba417ed7ea85c5e2bd7dacb4')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_tight.idName , '0437a4837f36ede6837c68e2369820b1')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_soft.idName  , 'b32cfe957062d29291b325ad34841caf')
central_id_registry.register(cutBasedMuonId_MuonPOG_V0_highpt.idName, '0cab98049677819d3e7ddf54ed3922f7')

