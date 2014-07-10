import FWCore.ParameterSet.Config as cms

from PhysicsTools.SelectorUtils.centralIDRegistry import central_id_registry

trivialCutFlow = cms.PSet(
    idName = cms.string("trivialCutFlow"),
    cutFlow = cms.VPSet(
        cms.PSet( cutName = cms.string("MinPtCut"),
                  minPt = cms.double(10.0),
                  isIsolation = cms.bool(False),
                  isIgnored = cms.bool(False)           ),
        cms.PSet( cutName = cms.string("MaxAbsEtaCut"),
                  maxEta = cms.double(2.5),
                  isIsolation = cms.bool(False),
                  isIgnored = cms.bool(False)           )
    )
)

central_id_registry.register(trivialCutFlow.idName,
                             'c819d8dbe312c620310afc2b253c5790')
