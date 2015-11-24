import FWCore.ParameterSet.Config as cms

RecoHiEvtPlaneFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*')
    )

RecoHiEvtPlaneRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*')
    )

RecoHiEvtPlaneAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoEvtPlanes_hiEvtPlane_*_*',
                                           'keep ZDCRecHitsSorted_zdcreco_*_*',
                                           #for backward compatibility
                                           'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                           #new place for ZDC
                                           'keep ZDCDataFramesSorted_castorDigis_*_*',
                                           'keep HFRecHitsSorted_hfreco_*_*'
                                          )
    )
