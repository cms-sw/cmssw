import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

trivialCutFlow = cms.PSet(
    idName = cms.string("trivialCutFlow"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MinPtCut"),
                  minPt = cms.double(10.0),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)                ),
        cms.PSet( cutName = cms.string("MaxAbsEtaCut"),
                  maxEta = cms.double(2.5),
                  needsAdditionalProducts = cms.bool(False),
                  isIgnored = cms.bool(False)                )
    )
)

central_id_registry.register(trivialCutFlow.idName,
                             '406a42716bb40f14256446a98e25c1de')
