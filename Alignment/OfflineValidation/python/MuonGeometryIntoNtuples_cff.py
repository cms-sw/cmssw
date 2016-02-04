import FWCore.ParameterSet.Config as cms

MuonGeometryIntoNtupleNoDB = cms.PSet(
    DTApplyAlignment = cms.bool(False),
    CSCApplyAlignment = cms.bool(False),
    DTAlignmentLabel = cms.string(''),
    CSCAlignmentLabel = cms.string(''),
    DTFromSurveyRcd = cms.bool(False),
    CSCFromSurveyRcd = cms.bool(False)
)
MuonGeometryIntoNtupleDefaultDB = cms.PSet(
    DTApplyAlignment = cms.bool(True),
    CSCApplyAlignment = cms.bool(True),
    DTAlignmentLabel = cms.string(''),
    CSCAlignmentLabel = cms.string(''),
    DTFromSurveyRcd = cms.bool(False),
    CSCFromSurveyRcd = cms.bool(False)
)
MuonGeometryIntoNtupleNoScenario = cms.PSet(
    MisalignmentScenario = cms.PSet(

    ),
    ApplyMisalignmentScenario = cms.bool(False)
)
MuonGeometryIntoNtuplesNoDT = cms.PSet(
    DTSuperLayers = cms.untracked.bool(False),
    DTChambers = cms.untracked.bool(False),
    DTWheels = cms.untracked.bool(False),
    DTStations = cms.untracked.bool(False),
    DTLayers = cms.untracked.bool(False)
)
MuonGeometryIntoNtuplesNoCSC = cms.PSet(
    CSCLayers = cms.untracked.bool(False),
    CSCChambers = cms.untracked.bool(False),
    CSCStations = cms.untracked.bool(False)
)

