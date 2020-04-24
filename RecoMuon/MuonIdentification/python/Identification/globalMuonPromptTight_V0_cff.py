import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

globalMuonPromptTight_V0 = cms.PSet(
    idName = cms.string("globalMuonPromptTight-V0"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("GlobalMuonPromptTightCut"),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)                )
        )
)

central_id_registry.register(globalMuonPromptTight_V0,
                             '4e200c57c24487498e1673f21c0e682e')
