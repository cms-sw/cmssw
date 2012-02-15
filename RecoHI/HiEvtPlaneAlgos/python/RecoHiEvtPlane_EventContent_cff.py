import FWCore.ParameterSet.Config as cms

RecoHiEvtPlaneFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*',
                                           'keep *_hiEvtPlaneFlat_*_*',
                                           )
    )

RecoHiEvtPlaneRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*',
                                           'keep *_hiEvtPlaneFlat_*_*',                                           
                                           )
    )

RecoHiEvtPlaneAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*',
                                           'keep *_hiEvtPlaneFlat_*_*',                                           
                                           )
    )
