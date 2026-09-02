import FWCore.ParameterSet.Config as cms

# AOD content
RecoHiEvtPlaneAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
	'keep recoEvtPlanes_hiEvtPlane_*_*',
        'keep ZDCRecHitsSorted_zdcreco_*_*',
        'keep ZDCDataFramesSorted_hcalDigis_*_*',
        'keep HFRecHitsSorted_hfreco_*_*')
)

# RECO content
RecoHiEvtPlaneRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoHiEvtPlaneRECO.outputCommands.extend(RecoHiEvtPlaneAOD.outputCommands)

# FEVT content
RecoHiEvtPlaneFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RecoHiEvtPlaneFEVT.outputCommands.extend(RecoHiEvtPlaneRECO.outputCommands)
