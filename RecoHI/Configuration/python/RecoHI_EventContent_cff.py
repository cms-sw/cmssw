import FWCore.ParameterSet.Config as cms

# include event content from RecoHI packages
from RecoHI.HiTracking.RecoHiTracker_EventContent_cff import *
from RecoHI.HiJetAlgos.RecoHiJets_EventContent_cff import *
from RecoHI.HiEgammaAlgos.RecoHiEgamma_EventContent_cff import *
from RecoHI.HiCentralityAlgos.RecoHiCentrality_EventContent_cff import *
from RecoHI.HiEvtPlaneAlgos.RecoHiEvtPlane_EventContent_cff import *
from RecoHI.HiMuonAlgos.RecoHiMuon_EventContent_cff import *

# combine RECO, AOD, FEVT content from all RecoHI packages
# RecoHI event contents to be included by Configuration.EventContent.EventContentHeavyIons_cff

RecoHIRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

RecoHIAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

RecoHIFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring()
    )

RecoHIRECO.outputCommands.extend(RecoHiTrackerRECO.outputCommands)
RecoHIRECO.outputCommands.extend(RecoHiTrackerLocalRECO.outputCommands)
RecoHIRECO.outputCommands.extend(RecoHiJetsRECO.outputCommands)
RecoHIRECO.outputCommands.extend(RecoHiEgammaRECO.outputCommands)
RecoHIRECO.outputCommands.extend(RecoHiEvtPlaneRECO.outputCommands)
RecoHIRECO.outputCommands.extend(RecoHiCentralityRECO.outputCommands)
RecoHIRECO.outputCommands.extend(RecoHiMuonRECO.outputCommands)

RecoHIAOD.outputCommands.extend(RecoHiTrackerAOD.outputCommands)
RecoHIAOD.outputCommands.extend(RecoHiJetsRECO.outputCommands)
RecoHIAOD.outputCommands.extend(RecoHiEgammaAOD.outputCommands)
RecoHIAOD.outputCommands.extend(RecoHiEvtPlaneAOD.outputCommands)
RecoHIAOD.outputCommands.extend(RecoHiCentralityAOD.outputCommands)
RecoHIAOD.outputCommands.extend(RecoHiMuonAOD.outputCommands)

RecoHIFEVT.outputCommands.extend(RecoHiTrackerFEVT.outputCommands)
RecoHIFEVT.outputCommands.extend(RecoHiTrackerLocalFEVT.outputCommands)
RecoHIFEVT.outputCommands.extend(RecoHiJetsFEVT.outputCommands)
RecoHIFEVT.outputCommands.extend(RecoHiEgammaFEVT.outputCommands)
RecoHIFEVT.outputCommands.extend(RecoHiEvtPlaneFEVT.outputCommands)
RecoHIFEVT.outputCommands.extend(RecoHiCentralityFEVT.outputCommands)
RecoHIFEVT.outputCommands.extend(RecoHiMuonFEVT.outputCommands)
