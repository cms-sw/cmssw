import FWCore.ParameterSet.Config as cms

RecoHiEvtPlaneFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*')
    )

RecoHiEvtPlaneRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*')
    )

RecoHiEvtPlaneAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*')
    )
