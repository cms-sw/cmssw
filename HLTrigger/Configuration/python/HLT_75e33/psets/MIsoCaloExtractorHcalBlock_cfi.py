import FWCore.ParameterSet.Config as cms

MIsoCaloExtractorHcalBlock = cms.PSet(
    CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
    ComponentName = cms.string('CaloExtractor'),
    DR_Max = cms.double(1.0),
    DR_Veto_E = cms.double(0.07),
    DR_Veto_H = cms.double(0.1),
    DepositLabel = cms.untracked.string('EcalPlusHcal'),
    Threshold_E = cms.double(0.2),
    Threshold_H = cms.double(0.5),
    Vertex_Constraint_XY = cms.bool(False),
    Vertex_Constraint_Z = cms.bool(False),
    Weight_E = cms.double(0.0),
    Weight_H = cms.double(1.0)
)