import FWCore.ParameterSet.Config as cms

#
#
# Event Content definition
#
# Data Tiers defined:
#
#  FEVT, RECO, AOD: 
#    include reconstruction content
#
#  FEVTSIM, RECOSIM, AODSIM: 
#    include reconstruction and simulation
#
#  FEVTSIMANA, RECOSIMANA, AODSIMANA: 
#    include reconstruction, simulation and analysis
#  FEVTSIMDIGIHLTDEBUG FEVTSIMHLTDEBUG
#
#  $Id: EventContent.cff,v 1.28 2008/04/18 04:18:44 dlange Exp $
#
#
#
#
# Recontruction Systems
#
#
from RecoLocalTracker.Configuration.RecoLocalTracker_EventContent_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuon_EventContent_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_EventContent_cff import *
from RecoEcal.Configuration.RecoEcal_EventContent_cff import *
from TrackingTools.Configuration.TrackingTools_EventContent_cff import *
from RecoTracker.Configuration.RecoTracker_EventContent_cff import *
from RecoJets.Configuration.RecoJets_EventContent_cff import *
from RecoMET.Configuration.RecoMET_EventContent_cff import *
from RecoMuon.Configuration.RecoMuon_EventContent_cff import *
from RecoBTau.Configuration.RecoBTau_EventContent_cff import *
from RecoBTag.Configuration.RecoBTag_EventContent_cff import *
from RecoTauTag.Configuration.RecoTauTag_EventContent_cff import *
from RecoVertex.Configuration.RecoVertex_EventContent_cff import *
from RecoPixelVertexing.Configuration.RecoPixelVertexing_EventContent_cff import *
from RecoEgamma.Configuration.RecoEgamma_EventContent_cff import *
from RecoParticleFlow.Configuration.RecoParticleFlow_EventContent_cff import *
from L1Trigger.Configuration.L1Trigger_EventContent_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_EventContent_cff import *
#DigiToRaw content
from EventFilter.Configuration.DigiToRaw_EventContent_cff import *
#
#
# Simulation Systems
#
#
from GeneratorInterface.Configuration.GeneratorInterface_EventContent_cff import *
from SimG4Core.Configuration.SimG4Core_EventContent_cff import *
from SimTracker.Configuration.SimTracker_EventContent_cff import *
from SimMuon.Configuration.SimMuon_EventContent_cff import *
from SimCalorimetry.Configuration.SimCalorimetry_EventContent_cff import *
from SimGeneral.Configuration.SimGeneral_EventContent_cff import *
from IOMC.RandomEngine.IOMC_EventContent_cff import *
#
#
# HLT
#
#
from HLTrigger.Configuration.HLTrigger_EventContent_cff import *
#
#
# Analysis Systems
#
#
from ElectroWeakAnalysis.Configuration.ElectroWeakAnalysis_EventContent_cff import *
from HiggsAnalysis.Configuration.HiggsAnalysis_EventContent_cff import *
from TopQuarkAnalysis.Configuration.TopQuarkAnalysis_EventContent_cff import *
#
#
# DQM
#
#
from DQMOffline.Configuration.DQMOffline_EventContent_cff import *
#
# FEVT Data Tier definition
#
#
FEVTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# RECO Data Tier definition
#
#
RECOEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# AOD Data Tier definition
#
#
AODEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# FEVTSIM Data Tier definition
#
#
FEVTSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# FEVTSIMDIGI Data Tier definition
#
#
FEVTSIMDIGIEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# RECOSIM Data Tier definition
#
#
RECOSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# AODSIM Data Tier definition
#
#
AODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# FEVTSIMANA Data Tier definition
#
#
FEVTSIMANAEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# FEVTSIMDIGIANA Data Tier definition
#
#
FEVTSIMDIGIANAEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# RECOSIMANA Data Tier definition
#
#
RECOSIMANAEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#
#
# AODSIMANA Data Tier definition
#
#
AODSIMANAEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
# RAW only data tier
RAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep  FEDRawDataCollection_rawDataCollector_*_*')
)
# 
FEVTSIMHLTDEBUGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
FEVTSIMDIGIHLTDEBUGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
FEVTEventContent.outputCommands.extend(RecoLocalTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalMuonFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalCaloFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoEcalFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(TrackingToolsFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoJetsFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoMETFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoMuonFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoBTauFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoBTagFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoTauTagFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoVertexFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoEgammaFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoPixelVertexingFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoParticleFlowFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(BeamSpotFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(L1TriggerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(HLTriggerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoEcalRECO.outputCommands)
RECOEventContent.outputCommands.extend(TrackingToolsRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoJetsRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoMETRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoMuonRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoBTauRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoBTagRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoTauTagRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoVertexRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoEgammaRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoPixelVertexingRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoParticleFlowRECO.outputCommands)
RECOEventContent.outputCommands.extend(BeamSpotRECO.outputCommands)
RECOEventContent.outputCommands.extend(L1TriggerRECO.outputCommands)
RECOEventContent.outputCommands.extend(HLTriggerRECO.outputCommands)
RECOEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalMuonAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalCaloAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoEcalAOD.outputCommands)
AODEventContent.outputCommands.extend(TrackingToolsAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoJetsAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoMETAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoMuonAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoBTauAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoBTagAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoTauTagAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoVertexAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoEgammaAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoParticleFlowAOD.outputCommands)
AODEventContent.outputCommands.extend(BeamSpotAOD.outputCommands)
AODEventContent.outputCommands.extend(L1TriggerAOD.outputCommands)
AODEventContent.outputCommands.extend(HLTriggerAOD.outputCommands)
AODEventContent.outputCommands.extend(MEtoEDMConverterAOD.outputCommands)
FEVTSIMEventContent.outputCommands.extend(FEVTEventContent.outputCommands)
FEVTSIMEventContent.outputCommands.extend(GeneratorInterfaceFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimG4CoreFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimTrackerFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimMuonFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimCalorimetryFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimGeneralFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(IOMCFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(FEVTEventContent.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(GeneratorInterfaceFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(SimG4CoreFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(SimGeneralFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(SimTrackerFEVTDIGI.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(SimMuonFEVTDIGI.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(SimCalorimetryFEVTDIGI.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(L1TriggerFEVTDIGI.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
FEVTSIMDIGIEventContent.outputCommands.extend(IOMCFEVT.outputCommands)
RECOSIMEventContent.outputCommands.extend(RECOEventContent.outputCommands)
RECOSIMEventContent.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimG4CoreRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimMuonRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimCalorimetryRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoGenMETRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoGenJetsRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimGeneralRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
AODSIMEventContent.outputCommands.extend(AODEventContent.outputCommands)
AODSIMEventContent.outputCommands.extend(GeneratorInterfaceAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimG4CoreAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimTrackerAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimMuonAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimCalorimetryAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoGenJetsAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoGenMETAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimGeneralAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(MEtoEDMConverterAOD.outputCommands)
FEVTSIMANAEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMANAEventContent.outputCommands.extend(ElectroWeakAnalysisEventContent.outputCommands)
FEVTSIMANAEventContent.outputCommands.extend(HiggsAnalysisEventContent.outputCommands)
FEVTSIMANAEventContent.outputCommands.extend(TopQuarkAnalysisEventContent.outputCommands)
FEVTSIMDIGIANAEventContent.outputCommands.extend(FEVTSIMDIGIEventContent.outputCommands)
FEVTSIMDIGIANAEventContent.outputCommands.extend(ElectroWeakAnalysisEventContent.outputCommands)
FEVTSIMDIGIANAEventContent.outputCommands.extend(HiggsAnalysisEventContent.outputCommands)
FEVTSIMDIGIANAEventContent.outputCommands.extend(TopQuarkAnalysisEventContent.outputCommands)
RECOSIMANAEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMANAEventContent.outputCommands.extend(ElectroWeakAnalysisEventContent.outputCommands)
RECOSIMANAEventContent.outputCommands.extend(HiggsAnalysisEventContent.outputCommands)
RECOSIMANAEventContent.outputCommands.extend(TopQuarkAnalysisEventContent.outputCommands)
AODSIMANAEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMANAEventContent.outputCommands.extend(ElectroWeakAnalysisEventContent.outputCommands)
AODSIMANAEventContent.outputCommands.extend(HiggsAnalysisEventContent.outputCommands)
AODSIMANAEventContent.outputCommands.extend(TopQuarkAnalysisEventContent.outputCommands)
FEVTSIMHLTDEBUGEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMHLTDEBUGEventContent.outputCommands.extend(HLTDebugFEVT.outputCommands)
FEVTSIMDIGIHLTDEBUGEventContent.outputCommands.extend(FEVTSIMDIGIEventContent.outputCommands)
FEVTSIMDIGIHLTDEBUGEventContent.outputCommands.extend(HLTDebugFEVT.outputCommands)

