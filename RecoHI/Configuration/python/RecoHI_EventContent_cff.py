import FWCore.ParameterSet.Config as cms

# include standard file
from Configuration.EventContent.EventContent_cff import *

# include from RecoHI packages
from RecoHI.HiTracking.RecoHiTracker_EventContent_cff import *
from RecoHI.HiJetAlgos.RecoHiJets_EventContent_cff import *
from RecoHI.HiEgammaAlgos.RecoHiEgamma_EventContent_cff import *
from RecoHI.HiCentralityAlgos.RecoHiCentrality_EventContent_cff import *
from RecoHI.HiEvtPlaneAlgos.RecoHiEvtPlane_EventContent_cff import *
#from RecoHI.HiMuonAlgos.RecoHiMuon_EventContent_cff import *

########################################################################
#
#  RAW , RECO, AOD: 
#    include reconstruction content
#
#  RAWSIM, RECOSIM, AODSIM: 
#    include reconstruction and simulation
#
#  RAWDEBUG(RAWSIM+ALL_SIM_INFO), RAWDEBUGHLT(RAWDEBUG+HLTDEBUG)
#
#  FEVT (RAW+RECO), FEVTSIM (RAWSIM+RECOSIM), 
#  FEVTDEBUG (FEVTSIM+ALL_SIM_INFO), FEVTDEBUGHLT (FEVTDEBUG+HLTDEBUG)
#
########################################################################

# extend existing data tiers with HI-specific content
RECOEventContent.outputCommands.extend(RecoHiTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoHiJetsRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoHiEgammaRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoHiEvtPlaneRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoHiCentralityRECO.outputCommands)

RECOSIMEventContent.outputCommands.extend(RecoHiTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoHiJetsRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoHiEgammaRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoHiEvtPlaneRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoHiCentralityRECO.outputCommands)

RECODEBUGEventContent.outputCommands.extend(RecoHiTrackerRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(RecoHiJetsRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(RecoHiEgammaRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(RecoHiEvtPlaneRECO.outputCommands)
RECODEBUGEventContent.outputCommands.extend(RecoHiCentralityRECO.outputCommands)


AODEventContent.outputCommands.extend(RecoHiTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoHiJetsRECO.outputCommands)
AODEventContent.outputCommands.extend(RecoHiEgammaAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoHiEvtPlaneAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoHiCentralityAOD.outputCommands)

AODSIMEventContent.outputCommands.extend(RecoHiTrackerAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoHiJetsRECO.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoHiEgammaAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoHiEvtPlaneAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoHiCentralityAOD.outputCommands)


FEVTEventContent.outputCommands.extend(RecoHiTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoHiJetsFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoHiEgammaFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoHiEvtPlaneFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoHiCentralityFEVT.outputCommands)

FEVTSIMEventContent.outputCommands.extend(RecoHiTrackerFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoHiJetsFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoHiEgammaFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoHiEvtPlaneFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoHiCentralityFEVT.outputCommands)

FEVTDEBUGEventContent.outputCommands.extend(RecoHiTrackerFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(RecoHiJetsFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(RecoHiEgammaFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(RecoHiEvtPlaneFEVT.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(RecoHiCentralityFEVT.outputCommands)
