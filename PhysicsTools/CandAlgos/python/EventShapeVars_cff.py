import FWCore.ParameterSet.Config as cms
import copy

caloEventShapeVars = cms.EDProducer("EventShapeVarsProducer",
    src = cms.InputTag("towerMaker"),
    r = cms.double(2.)
)

pfEventShapeVars = copy.deepcopy(caloEventShapeVars)
pfEventShapeVars.src = cms.InputTag("pfNoPileUp")

produceEventShapeVars = cms.Sequence( caloEventShapeVars * pfEventShapeVars )
