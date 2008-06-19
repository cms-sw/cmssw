# The following comments couldn't be translated into the new config version:

#replace FEVTEventContent.outputCommands += TrackingToolsFEVT.outputCommands

#replace FEVTEventContent.outputCommands += RecoBTauFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoBTagFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoTauTagFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoVertexFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoEgammaFEVT.outputCommands
#replace FEVTEventContent.outputCommands += RecoPixelVertexingFEVT.outputCommands

#replace FEVTEventContent.outputCommands += L1TriggerFEVT.outputCommands 

#replace RECOEventContent.outputCommands += TrackingToolsRECO.outputCommands

#replace RECOEventContent.outputCommands += RecoBTauRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoBTagRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoTauTagRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoVertexRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoEgammaRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoPixelVertexingRECO.outputCommands
#replace RECOEventContent.outputCommands += RecoParticleFlowRECO.outputCommands

#replace AODEventContent.outputCommands += TrackingToolsAOD.outputCommands

#replace AODEventContent.outputCommands += RecoBTauAOD.outputCommands
#replace AODEventContent.outputCommands += RecoBTagAOD.outputCommands
#replace AODEventContent.outputCommands += RecoTauTagAOD.outputCommands
#replace AODEventContent.outputCommands += RecoVertexAOD.outputCommands
#replace AODEventContent.outputCommands += RecoEgammaAOD.outputCommands
#replace AODEventContent.outputCommands += RecoParticleFlowAOD.outputCommands

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
#  $Id: EventContentCosmics.cff,v 1.1 2008/06/09 08:36:35 arizzi Exp $
#
#
#
#
# Recontruction Systems
#
#
from RecoTracker.Configuration.RecoTrackerP5_EventContent_cff import *
from RecoMuon.Configuration.RecoMuonCosmics_EventContent_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuonCosmics_EventContent_cff import *
from RecoEcal.Configuration.RecoEcal_EventContentCosmics_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_EventContentCosmics_cff import *
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_EventContent_cff import *
from RecoJets.Configuration.RecoJets_EventContent_cff import *
from RecoMET.Configuration.RecoMET_EventContent_cff import *
from L1Trigger.Configuration.L1Trigger_EventContent_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_EventContent_cff import *
from DQMOffline.Configuration.DQMOffline_EventContent_cff import *
from HLTrigger.Configuration.HLTrigger_EventContent_cff import *
from GeneratorInterface.Configuration.GeneratorInterface_EventContent_cff import *
from SimG4Core.Configuration.SimG4Core_EventContent_cff import *
from SimTracker.Configuration.SimTracker_EventContent_cff import *
from SimMuon.Configuration.SimMuon_EventContent_cff import *
from SimCalorimetry.Configuration.SimCalorimetry_EventContent_cff import *
from SimGeneral.Configuration.SimGeneral_EventContent_cff import *
from IOMC.RandomEngine.IOMC_EventContent_cff import *
from EventFilter.Configuration.DigiToRaw_EventContent_cff import *
#not in GR
#include "TrackingTools/Configuration/data/TrackingTools_EventContent.cff"
#include "RecoBTau/Configuration/data/RecoBTau_EventContent.cff"
#include "RecoBTag/Configuration/data/RecoBTag_EventContent.cff"
#include "RecoTauTag/Configuration/data/RecoTauTag_EventContent.cff"
#include "RecoVertex/Configuration/data/RecoVertex_EventContent.cff"
#include "RecoPixelVertexing/Configuration/data/RecoPixelVertexing_EventContent.cff"
#include "RecoEgamma/Configuration/data/RecoEgamma_EventContent.cff"
#include "RecoParticleFlow/Configuration/data/RecoParticleFlow_EventContent.cff"
#
# FEVT Data Tier definition
#
#
FEVTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *')
)
#replace FEVTEventContent.outputCommands += HLTriggerFEVT.outputCommands 
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
# RAW only data tier
RAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep  FEDRawDataCollection_rawDataCollector_*_*', 
        'keep  FEDRawDataCollection_source_*_*')
)
#
#
# RAWSIM Data Tier definition
#
#
RAWSIMEventContent = cms.PSet(
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
FEVTEventContent.outputCommands.extend(RecoLocalTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalMuonFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalCaloFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoEcalFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoTrackerFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoJetsFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoMETFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(RecoMuonFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(BeamSpotFEVT.outputCommands)
FEVTEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoEcalRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoJetsRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoMETRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoMuonRECO.outputCommands)
RECOEventContent.outputCommands.extend(BeamSpotRECO.outputCommands)
RECOEventContent.outputCommands.extend(L1TriggerRECO.outputCommands)
RECOEventContent.outputCommands.extend(HLTriggerRECO.outputCommands)
RECOEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalMuonAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalCaloAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoEcalAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoJetsAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoMETAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoMuonAOD.outputCommands)
AODEventContent.outputCommands.extend(BeamSpotAOD.outputCommands)
AODEventContent.outputCommands.extend(MEtoEDMConverterAOD.outputCommands)
RAWEventContent.outputCommands.extend(L1TriggerRAW.outputCommands)
RAWEventContent.outputCommands.extend(HLTriggerRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(RAWEventContent.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimG4CoreRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimTrackerRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimMuonRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimCalorimetryRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimGeneralRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(GeneratorInterfaceRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(IOMCRAW.outputCommands)
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

