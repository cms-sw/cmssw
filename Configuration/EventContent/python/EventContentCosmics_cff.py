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
#  $Id: EventContentCosmics_cff.py,v 1.22 2010/09/08 13:29:39 vlimant Exp $
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
from RecoVertex.BeamSpotProducer.BeamSpot_EventContent_cff import *
from RecoEgamma.Configuration.RecoEgamma_EventContent_cff import *
from RecoVertex.Configuration.RecoVertex_EventContent_cff import *
# raw2digi that are already the final RECO/AOD products
from EventFilter.ScalersRawToDigi.Scalers_EventContent_cff import *
from EventFilter.OnlineMetaDataRawToDigi.OnlineMetaData_EventContent_cff import *
from EventFilter.OnlineMetaDataRawToDigi.Tcds_EventContent_cff import *
# DigiToRaw content
from EventFilter.Configuration.DigiToRaw_EventContent_cff import *
# Simulation System
from GeneratorInterface.Configuration.GeneratorInterface_EventContent_cff import *
from SimG4Core.Configuration.SimG4Core_EventContent_cff import *
from SimTracker.Configuration.SimTracker_EventContent_cff import *
from SimMuon.Configuration.SimMuon_EventContent_cff import *
from SimCalorimetry.Configuration.SimCalorimetry_EventContent_cff import *
from SimGeneral.Configuration.SimGeneral_EventContent_cff import *
from IOMC.RandomEngine.IOMC_EventContent_cff import *
# L1
from L1Trigger.Configuration.L1Trigger_EventContent_Cosmics_cff import *
# HLT
from HLTrigger.Configuration.HLTrigger_EventContent_cff import *
# DQM
from DQMOffline.Configuration.DQMOffline_EventContent_cff import *

from Configuration.EventContent.EventContent_cff import REGENEventContent,RESIMEventContent,REDIGIEventContent
from Configuration.EventContent.EventContent_cff import DQMEventContent

#not in Cosmics 
#include "TrackingTools/Configuration/data/TrackingTools_EventContent.cff"
#include "RecoBTau/Configuration/data/RecoBTau_EventContent.cff"
#include "RecoBTag/Configuration/data/RecoBTag_EventContent.cff"
#include "RecoTauTag/Configuration/data/RecoTauTag_EventContent.cff"
#include "RecoVertex/Configuration/data/RecoVertex_EventContent.cff"
#include "RecoTracker/Configuration/data/RecoPixelVertexing_EventContent.cff"
#include "RecoEgamma/Configuration/data/RecoEgamma_EventContent.cff"
#include "RecoParticleFlow/Configuration/data/RecoParticleFlow_EventContent.cff"

# RAW only data tier
RAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep  FEDRawDataCollection_rawDataCollector_*_*', 
        'keep  FEDRawDataCollection_source_*_*'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
RAWEventContent.outputCommands.extend(L1TriggerRAW.outputCommands)
RAWEventContent.outputCommands.extend(HLTriggerRAW.outputCommands)
#
#
# RECO Data Tier definition
#
#
RECOEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_logErrorHarvester_*_*'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
RECOEventContent.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoEcalRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoEgammaRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoTrackerRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoJetsRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoMETRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoMuonRECO.outputCommands)
RECOEventContent.outputCommands.extend(BeamSpotRECO.outputCommands)
RECOEventContent.outputCommands.extend(RecoVertexRECO.outputCommands)
RECOEventContent.outputCommands.extend(L1TriggerRECO.outputCommands)
RECOEventContent.outputCommands.extend(HLTriggerRECO.outputCommands)
RECOEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
RECOEventContent.outputCommands.extend(EvtScalersRECO.outputCommands)
RECOEventContent.outputCommands.extend(OnlineMetaDataContent.outputCommands)
RECOEventContent.outputCommands.extend(TcdsEventContent.outputCommands)
#
#
# AOD Data Tier definition
#
#
AODEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_logErrorHarvester_*_*'),
    eventAutoFlushCompressedSize=cms.untracked.int32(15*1024*1024)
)
AODEventContent.outputCommands.extend(RecoLocalTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalMuonAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoLocalCaloAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoEcalAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoEgammaAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoTrackerAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoJetsAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoMETAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoMuonAOD.outputCommands)
AODEventContent.outputCommands.extend(BeamSpotAOD.outputCommands)
AODEventContent.outputCommands.extend(RecoVertexAOD.outputCommands)
AODEventContent.outputCommands.extend(MEtoEDMConverterAOD.outputCommands)
AODEventContent.outputCommands.extend(EvtScalersAOD.outputCommands)
AODEventContent.outputCommands.extend(OnlineMetaDataContent.outputCommands)
AODEventContent.outputCommands.extend(TcdsEventContent.outputCommands)
#
#
# RAWSIM Data Tier definition
#
#
RAWSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
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
#
#
# RECOSIM Data Tier definition
#
#
RECOSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_logErrorHarvester_*_*'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
RECOSIMEventContent.outputCommands.extend(RECOEventContent.outputCommands)
RECOSIMEventContent.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimG4CoreRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimMuonRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimCalorimetryRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoGenMETRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoGenJetsRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimGeneralRECO.outputCommands)
#
# FEVT Data Tier definition
#
#
#FEVT is by definition RECO + RAW
FEVTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_logErrorHarvester_*_*'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
FEVTEventContent.outputCommands.extend(RAWEventContent.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalMuonRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoLocalCaloRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoEcalRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoEgammaRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoTrackerRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoJetsRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoMETRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoMuonRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(BeamSpotRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(RecoVertexRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(L1TriggerRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(HLTriggerRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(EvtScalersRECO.outputCommands)	
FEVTEventContent.outputCommands.extend(OnlineMetaDataContent.outputCommands)	
FEVTEventContent.outputCommands.extend(TcdsEventContent.outputCommands)	

#replace FEVTEventContent.outputCommands += HLTriggerFEVT.outputCommands 
FEVTHLTALLEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
FEVTHLTALLEventContent.outputCommands.extend(FEVTEventContent.outputCommands)
FEVTHLTALLEventContent.outputCommands.append('keep *_*_*_HLT')
#
# FEVTSIM Data Tier definition
#
#
FEVTSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_logErrorHarvester_*_*'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
FEVTSIMEventContent.outputCommands.extend(FEVTEventContent.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimG4CoreRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimTrackerRAW.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimMuonRAW.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimCalorimetryRAW.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimGeneralRAW.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(GeneratorInterfaceRAW.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(IOMCRAW.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(RecoGenMETRECO.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(RecoGenJetsRECO.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimG4CoreRECO.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimTrackerRECO.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimMuonRECO.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimCalorimetryRECO.outputCommands)	
FEVTSIMEventContent.outputCommands.extend(SimGeneralRECO.outputCommands)
#
#
# FEVTDEBUG Data Tier definition
#
#
FEVTDEBUGEventContent = cms.PSet(
   outputCommands = cms.untracked.vstring('drop *',
       'keep *_logErrorHarvester_*_*'),
   splitLevel = cms.untracked.int32(0),
   eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)
FEVTDEBUGEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(L1TriggerFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimGeneralFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimTrackerFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimMuonFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimCalorimetryFEVTDEBUG.outputCommands)
#
#
# ALCARECO Data Tier definition
#
#
