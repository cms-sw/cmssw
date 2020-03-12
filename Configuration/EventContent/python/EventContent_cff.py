import FWCore.ParameterSet.Config as cms

#
# Event Content definition
#
# Data Tiers defined:
#
#  LHE:
#    include pure LHE production
#
#  RAW , RECO, AOD:
#    include reconstruction content
#
#  RAWSIM, RECOSIM, AODSIM:
#    include reconstruction and simulation
#
#  GENRAW
#    slimmed-down version of RAWSIM for small transient disk size during MC production, contains Gen+Rawdata
#
#  PREMIX
#    contains special Digi collection(s) for pre-mixing minbias events for pileup simulation
#    Raw2Digi step is done on this file.
#
#  PREMIXRAW
#    extension of RAWSIM for output of second stage of PreMixing using the DataMixer.  
#
#  RAWDEBUG(RAWSIM+ALL_SIM_INFO), RAWDEBUGHLT(RAWDEBUG+HLTDEBUG)
#
#  RAWSIMHLT (RAWSIM + HLTDEBUG)
#
#  RAWRECOSIMHLT, RAWRECODEBUGHLT
#
#  FEVT (RAW+RECO), FEVTSIM (RAWSIM+RECOSIM), FEVTDEBUG (FEVTSIM+ALL_SIM_INFO), FEVTDEBUGHLT (FEVTDEBUG+HLTDEBUG)
#
#  $Id: EventContent_cff.py,v 1.54 2013/05/01 15:44:29 mikeh Exp $
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
from CommonTools.ParticleFlow.EITopPAG_EventContent_cff import EITopPAGEventContent
from RecoPPS.Configuration.RecoCTPPS_EventContent_cff import *

# raw2digi that are already the final RECO/AOD products
from EventFilter.ScalersRawToDigi.Scalers_EventContent_cff import *
from EventFilter.OnlineMetaDataRawToDigi.OnlineMetaData_EventContent_cff import *
from EventFilter.Utilities.Tcds_EventContent_cff import *

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
from SimFastTiming.Configuration.SimFastTiming_EventContent_cff import *
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
# DQM
#
#
from DQMOffline.Configuration.DQMOffline_EventContent_cff import *
#
#
# NANOAOD
#
#
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *


#
#
# FastSim
#
#
from FastSimulation.Configuration.EventContent_cff import FASTPUEventContent
import FastSimulation.Configuration.EventContent_cff as fastSimEC
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(RecoLocalTrackerRECO, outputCommands = fastSimEC.RecoLocalTracker.outputCommands)
fastSim.toModify(RecoLocalTrackerFEVT, outputCommands = fastSimEC.RecoLocalTracker.outputCommands)
fastSim.toReplaceWith(SimG4CoreRAW, fastSimEC.SimRAW)
fastSim.toReplaceWith(SimG4CoreRECO, fastSimEC.SimRECO)

#
#
# Top level additional keep statements
#
#
CommonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_logErrorHarvester_*_*')
)

#
#
# LHE Data Tier definition
#
#
LHEEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# RAW Data Tier definition
#
#
RAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
        'keep  FEDRawDataCollection_rawDataCollector_*_*',
        'keep  FEDRawDataCollection_source_*_*'),
    splitLevel = cms.untracked.int32(0),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)
#
#
# RECO Data Tier definition
#
#
RECOEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# RAWRECO Data Tier definition
#
#
RAWRECOEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
    )
#
#
# AOD Data Tier definition
#
#
AODEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(30*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)
#
#
# RAWAOD Data Tier definition
#
#
RAWAODEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(30*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)
#
#
# RAWSIM Data Tier definition
# ===========================
#
# Here, we sacrifice memory and CPU time to decrease the on-disk size as
# much as possible.  Given the current per-event GEN-SIM and DIGI-RECO times,
# the extra CPU time for LZMA compression works out to be ~1%.  The GEN-SIM
# use case of reading a minbias event for `classic pileup` has a similar CPU
# impact.
# The memory increase appears to be closer to 50MB - but that should be
# acceptable as the introduction of multithreaded processing has bought us some
# breathing room.
#
RAWSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(20*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(1)
)
#
#
# RAWSIMHLT Data Tier definition
#
#
RAWSIMHLTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# RAWRECOSIMHLT Data Tier definition
#
#
RAWRECOSIMHLTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# RAWRECODEBUGHLT Data Tier definition
#
#
RAWRECODEBUGHLTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# RECOSIM Data Tier definition
#
#
RECOSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# GENRAW Data Tier definition
#
#
GENRAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# AODSIM Data Tier definition
#
#
AODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(30*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4),
)
#
#
# FEVT Data Tier definition
#
#
FEVTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
FEVTHLTALLEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)

#
#
# FEVTSIM Data Tier definition
#
#
FEVTSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# RAWDEBUG Data Tier definition
#
#
RAWDEBUGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# RAWDEBUGHLT Data Tier definition
#
#
RAWDEBUGHLTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)
#
#
# FEVTDEBUG Data Tier definition
#
#
FEVTDEBUGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)


#
#
# FEVTDEBUGHLT Data Tier definition
#
#
FEVTDEBUGHLTEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)

#
#
# RECOSIMDEBUG Data Tier definition
#
#
RECODEBUGEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)


#
## HLTDEBUG tier definition
#
HLTDEBUGEventContent = cms.PSet(
    #outputCommands = cms.untracked.vstring('drop *',
    #        'keep *_hlt*_*_*')
    outputCommands = cms.untracked.vstring('drop *',
        'keep *_logErrorHarvester_*_*'),
    splitLevel = cms.untracked.int32(0),
)

#
#
## DQM event content
#
#
DQMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep *_MEtoEDMConverter_*_*'),
    splitLevel = cms.untracked.int32(0)
    )

#Special Event Content for MixingModule and DataMixer
DATAMIXEREventContent = cms.PSet(
        outputCommands = cms.untracked.vstring('drop *',
                                               'keep CSCDetIdCSCALCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCALCTDigi_*',
                                               'keep CSCDetIdCSCCLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCLCTDigi_*',
                                               'keep CSCDetIdCSCComparatorDigiMuonDigiCollection_muonCSCDigis_MuonCSCComparatorDigi_*',
                                               'keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_csctfDigis_*_*',
                                               'keep CSCDetIdCSCCorrelatedLCTDigiMuonDigiCollection_muonCSCDigis_MuonCSCCorrelatedLCTDigi_*',
                                               'keep CSCDetIdCSCRPCDigiMuonDigiCollection_muonCSCDigis_MuonCSCRPCDigi_*',
                                               'keep CSCDetIdCSCStripDigiMuonDigiCollection_muonCSCDigis_MuonCSCStripDigi_*',
                                               'keep CSCDetIdCSCWireDigiMuonDigiCollection_muonCSCDigis_MuonCSCWireDigi_*',
                                               'keep DTLayerIdDTDigiMuonDigiCollection_muonDTDigis_*_*',
                                               'keep PixelDigiedmDetSetVector_siPixelDigis_*_*',
                                               'keep SiStripDigiedmDetSetVector_siStripDigis_*_*',
                                               'keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*',
                                               'keep HBHEDataFramesSorted_hcalDigis_*_*',
                                               'keep HFDataFramesSorted_hcalDigis_*_*',
                                               'keep HODataFramesSorted_hcalDigis_*_*',
                                               'keep QIE10DataFrameHcalDataFrameContainer_hcalDigis_*_*',
                                               'keep QIE11DataFrameHcalDataFrameContainer_hcalDigis_*_*',
                                               'keep ZDCDataFramesSorted_hcalDigis_*_*',
                                               'keep CastorDataFramesSorted_castorDigis_*_*',
                                               'keep EBDigiCollection_ecalDigis_*_*',
                                               'keep EEDigiCollection_ecalDigis_*_*',
                                               'keep ESDigiCollection_ecalPreshowerDigis_*_*'),
        splitLevel = cms.untracked.int32(0),
        )

PREMIXEventContent = cms.PSet(
        outputCommands = cms.untracked.vstring('drop *'),
        splitLevel = cms.untracked.int32(0),
        )

MIXINGMODULEEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
                                           'keep *_cfWriter_*_*'),
    splitLevel = cms.untracked.int32(0),
    )

# PREMIXRAW Data Tier definition
#
#
PREMIXRAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    splitLevel = cms.untracked.int32(0),
)

#
#
## RAW repacked event content definition
#
#
REPACKRAWEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(
      'drop *',
      'drop FEDRawDataCollection_*_*_*',
      'keep FEDRawDataCollection_rawDataRepacker_*_*',
      'keep FEDRawDataCollection_virginRawDataRepacker_*_*',
      'keep FEDRawDataCollection_rawDataReducedFormat_*_*'),
    splitLevel = cms.untracked.int32(0),
    )
REPACKRAWSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring(),
    splitLevel = cms.untracked.int32(0),
)

LHEEventContent.outputCommands.extend(GeneratorInterfaceLHE.outputCommands)

HLTDEBUGEventContent.outputCommands.extend(HLTDebugFEVT.outputCommands)

RAWEventContent.outputCommands.extend(L1TriggerRAW.outputCommands)
RAWEventContent.outputCommands.extend(HLTriggerRAW.outputCommands)

REPACKRAWEventContent.outputCommands.extend(L1TriggerRAW.outputCommands)
REPACKRAWEventContent.outputCommands.extend(HLTriggerRAW.outputCommands)


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
RECOEventContent.outputCommands.extend(EvtScalersRECO.outputCommands)
RECOEventContent.outputCommands.extend(OnlineMetaDataContent.outputCommands)
RECOEventContent.outputCommands.extend(TcdsEventContent.outputCommands)
RECOEventContent.outputCommands.extend(CommonEventContent.outputCommands)
RECOEventContent.outputCommands.extend(EITopPAGEventContent.outputCommands)
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
ctpps_2016.toModify(RECOEventContent, outputCommands = RECOEventContent.outputCommands + RecoCTPPSRECO.outputCommands)

RAWRECOEventContent.outputCommands.extend(RECOEventContent.outputCommands)
RAWRECOEventContent.outputCommands.extend(cms.untracked.vstring(
    'keep FEDRawDataCollection_rawDataCollector_*_*',
    'keep FEDRawDataCollection_source_*_*'
))

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
AODEventContent.outputCommands.extend(EvtScalersAOD.outputCommands)
AODEventContent.outputCommands.extend(OnlineMetaDataContent.outputCommands)
AODEventContent.outputCommands.extend(TcdsEventContent.outputCommands)
AODEventContent.outputCommands.extend(CommonEventContent.outputCommands)
ctpps_2016.toModify(AODEventContent, outputCommands = AODEventContent.outputCommands + RecoCTPPSAOD.outputCommands)

RAWAODEventContent.outputCommands.extend(AODEventContent.outputCommands)
RAWAODEventContent.outputCommands.extend(cms.untracked.vstring(
    'keep FEDRawDataCollection_rawDataCollector_*_*',
    'keep FEDRawDataCollection_source_*_*'
))

RAWSIMEventContent.outputCommands.extend(RAWEventContent.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimG4CoreRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimTrackerRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimMuonRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimCalorimetryRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimFastTimingRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(SimGeneralRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(GeneratorInterfaceRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
RAWSIMEventContent.outputCommands.extend(IOMCRAW.outputCommands)
RAWSIMEventContent.outputCommands.extend(CommonEventContent.outputCommands)

RAWSIMHLTEventContent.outputCommands.extend(RAWSIMEventContent.outputCommands)
RAWSIMHLTEventContent.outputCommands.extend(HLTDebugRAW.outputCommands)

GENRAWEventContent.outputCommands.extend(RAWEventContent.outputCommands)
GENRAWEventContent.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)
GENRAWEventContent.outputCommands.extend(SimG4CoreRECO.outputCommands)
GENRAWEventContent.outputCommands.extend(SimTrackerRAW.outputCommands)
GENRAWEventContent.outputCommands.extend(SimMuonRECO.outputCommands)
GENRAWEventContent.outputCommands.extend(SimCalorimetryRECO.outputCommands)
GENRAWEventContent.outputCommands.extend(SimFastTimingRECO.outputCommands)
GENRAWEventContent.outputCommands.extend(SimGeneralRECO.outputCommands)
GENRAWEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)
GENRAWEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)
GENRAWEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
GENRAWEventContent.outputCommands.extend(IOMCRAW.outputCommands)
GENRAWEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)
GENRAWEventContent.outputCommands.extend(CommonEventContent.outputCommands)

PREMIXEventContent.outputCommands.extend(SimGeneralRAW.outputCommands)
PREMIXEventContent.outputCommands.extend(IOMCRAW.outputCommands)
PREMIXEventContent.outputCommands.extend(CommonEventContent.outputCommands)
PREMIXEventContent.outputCommands.extend(SimTrackerPREMIX.outputCommands)
PREMIXEventContent.outputCommands.extend(SimCalorimetryPREMIX.outputCommands)
PREMIXEventContent.outputCommands.extend(SimFastTimingPREMIX.outputCommands)
PREMIXEventContent.outputCommands.extend(SimMuonPREMIX.outputCommands)
PREMIXEventContent.outputCommands.extend(SimGeneralPREMIX.outputCommands)
fastSim.toModify(PREMIXEventContent, outputCommands = PREMIXEventContent.outputCommands+fastSimEC.extraPremixContent)

PREMIXRAWEventContent.outputCommands.extend(RAWSIMEventContent.outputCommands)
PREMIXRAWEventContent.outputCommands.append('keep CrossingFramePlaybackInfoNew_*_*_*')
PREMIXRAWEventContent.outputCommands.append('drop CrossingFramePlaybackInfoNew_mix_*_*')
PREMIXRAWEventContent.outputCommands.append('keep *_*_MergedTrackTruth_*')
PREMIXRAWEventContent.outputCommands.append('keep *_*_StripDigiSimLink_*')
PREMIXRAWEventContent.outputCommands.append('keep *_*_PixelDigiSimLink_*')
PREMIXRAWEventContent.outputCommands.append('keep *_*_MuonCSCStripDigiSimLinks_*')
PREMIXRAWEventContent.outputCommands.append('keep *_*_MuonCSCWireDigiSimLinks_*')
PREMIXRAWEventContent.outputCommands.append('keep *_*_RPCDigiSimLink_*')
PREMIXRAWEventContent.outputCommands.append('keep DTLayerIdDTDigiSimLinkMuonDigiCollection_*_*_*')
fastSim.toModify(PREMIXEventContent, outputCommands = PREMIXEventContent.outputCommands+fastSimEC.extraPremixContent)

REPACKRAWSIMEventContent.outputCommands.extend(REPACKRAWEventContent.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(SimG4CoreRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(SimTrackerRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(SimMuonRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(SimCalorimetryRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(SimFastTimingRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(SimGeneralRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(GeneratorInterfaceRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(IOMCRAW.outputCommands)
REPACKRAWSIMEventContent.outputCommands.extend(CommonEventContent.outputCommands)

RECOSIMEventContent.outputCommands.extend(RECOEventContent.outputCommands)
RECOSIMEventContent.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoGenMETRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(RecoGenJetsRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimG4CoreRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimTrackerRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimMuonRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimCalorimetryRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimFastTimingRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(SimGeneralRECO.outputCommands)
RECOSIMEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)

AODSIMEventContent.outputCommands.extend(AODEventContent.outputCommands)
AODSIMEventContent.outputCommands.extend(GeneratorInterfaceAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimG4CoreAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimTrackerAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimMuonAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimCalorimetryAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimFastTimingAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoGenJetsAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(RecoGenMETAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(SimGeneralAOD.outputCommands)
AODSIMEventContent.outputCommands.extend(MEtoEDMConverterAOD.outputCommands)

RAWRECOSIMHLTEventContent.outputCommands.extend(RAWRECOEventContent.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(RecoGenMETRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(RecoGenJetsRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(SimG4CoreRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(SimTrackerRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(SimMuonRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(SimCalorimetryRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(SimFastTimingRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(SimGeneralRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
RAWRECOSIMHLTEventContent.outputCommands.extend(HLTDebugRAW.outputCommands)

RAWRECODEBUGHLTEventContent.outputCommands.extend(RAWRECOSIMHLTEventContent.outputCommands)
RAWRECODEBUGHLTEventContent.outputCommands.extend(SimGeneralFEVTDEBUG.outputCommands)
RAWRECODEBUGHLTEventContent.outputCommands.extend(SimTrackerDEBUG.outputCommands)

FEVTEventContent.outputCommands.extend(RAWEventContent.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoEcalRECO.outputCommands)
FEVTEventContent.outputCommands.extend(TrackingToolsRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoTrackerRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoJetsRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoMETRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoMuonRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoBTauRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoBTagRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoTauTagRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoVertexRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoEgammaRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoPixelVertexingRECO.outputCommands)
FEVTEventContent.outputCommands.extend(RecoParticleFlowRECO.outputCommands)
FEVTEventContent.outputCommands.extend(BeamSpotRECO.outputCommands)
FEVTEventContent.outputCommands.extend(L1TriggerRECO.outputCommands)
FEVTEventContent.outputCommands.extend(HLTriggerRECO.outputCommands)
FEVTEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
FEVTEventContent.outputCommands.extend(EvtScalersRECO.outputCommands)
FEVTEventContent.outputCommands.extend(OnlineMetaDataContent.outputCommands)
FEVTEventContent.outputCommands.extend(TcdsEventContent.outputCommands)
FEVTEventContent.outputCommands.extend(CommonEventContent.outputCommands)
FEVTEventContent.outputCommands.extend(EITopPAGEventContent.outputCommands)
ctpps_2016.toModify(FEVTEventContent, outputCommands = FEVTEventContent.outputCommands + RecoCTPPSFEVT.outputCommands)

FEVTHLTALLEventContent.outputCommands.extend(FEVTEventContent.outputCommands)
FEVTHLTALLEventContent.outputCommands.append('keep *_*_*_HLT')


FEVTSIMEventContent.outputCommands.extend(RAWEventContent.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimG4CoreRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimTrackerRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimMuonRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimCalorimetryRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimFastTimingRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimGeneralRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(GeneratorInterfaceRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoGenJetsFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoGenMETFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(DigiToRawFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(MEtoEDMConverterFEVT.outputCommands)
FEVTSIMEventContent.outputCommands.extend(IOMCRAW.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoLocalTrackerRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoLocalMuonRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoLocalCaloRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoEcalRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(TrackingToolsRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoTrackerRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoJetsRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoMETRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoMuonRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoBTauRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoBTagRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoTauTagRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoVertexRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoEgammaRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoPixelVertexingRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoParticleFlowRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(BeamSpotRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(L1TriggerRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(HLTriggerRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(GeneratorInterfaceRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoGenMETRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(RecoGenJetsRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimG4CoreRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimTrackerRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimMuonRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimCalorimetryRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimFastTimingRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(SimGeneralRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(MEtoEDMConverterRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(EvtScalersRECO.outputCommands)
FEVTSIMEventContent.outputCommands.extend(CommonEventContent.outputCommands)
FEVTSIMEventContent.outputCommands.extend(EITopPAGEventContent.outputCommands)
FEVTSIMEventContent.outputCommands.extend(OnlineMetaDataContent.outputCommands)
FEVTSIMEventContent.outputCommands.extend(TcdsEventContent.outputCommands)
RAWDEBUGEventContent.outputCommands.extend(RAWSIMEventContent.outputCommands)
RAWDEBUGEventContent.outputCommands.extend(SimTrackerDEBUG.outputCommands)
RAWDEBUGEventContent.outputCommands.extend(SimGeneralFEVTDEBUG.outputCommands)
RAWDEBUGEventContent.outputCommands.extend(L1TriggerRAWDEBUG.outputCommands)
RAWDEBUGHLTEventContent.outputCommands.extend(RAWDEBUGEventContent.outputCommands)
RAWDEBUGHLTEventContent.outputCommands.extend(HLTDebugRAW.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(L1TriggerFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimGeneralFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimTrackerFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimMuonFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimCalorimetryFEVTDEBUG.outputCommands)
FEVTDEBUGEventContent.outputCommands.extend(SimFastTimingFEVTDEBUG.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(FEVTDEBUGEventContent.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.extend(HLTDebugFEVT.outputCommands)
FEVTDEBUGHLTEventContent.outputCommands.append('keep *_*_MergedTrackTruth_*')
FEVTDEBUGHLTEventContent.outputCommands.append('keep *_*_StripDigiSimLink_*')
FEVTDEBUGHLTEventContent.outputCommands.append('keep *_*_PixelDigiSimLink_*')
RECODEBUGEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECODEBUGEventContent.outputCommands.extend(SimGeneralFEVTDEBUG.outputCommands)
RECODEBUGEventContent.outputCommands.extend(SimTrackerDEBUG.outputCommands)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
(premix_stage2 & phase2_tracker).toModify(FEVTDEBUGHLTEventContent, outputCommands = FEVTDEBUGHLTEventContent.outputCommands+[
    'keep *_*_Phase2OTDigiSimLink_*'
])
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
(premix_stage2 & phase2_muon).toModify(FEVTDEBUGHLTEventContent, outputCommands = FEVTDEBUGHLTEventContent.outputCommands+[
    'keep *_*_GEMDigiSimLink_*',
    'keep *_*_GEMStripDigiSimLink_*',
    'keep *_*_ME0DigiSimLink_*',
    'keep *_*_ME0StripDigiSimLink_*',
])

REPACKRAWSIMEventContent.outputCommands.extend(['drop FEDRawDataCollection_source_*_*',
                                                'drop FEDRawDataCollection_rawDataCollector_*_*'])
REPACKRAWEventContent.outputCommands.extend(['drop FEDRawDataCollection_source_*_*',
                                                'drop FEDRawDataCollection_rawDataCollector_*_*'])



#from modules in Configuration.StandardSequence.Generator_cff fixGenInfo
REGENEventContent = cms.PSet(
    inputCommands=cms.untracked.vstring(
      'keep *',
      'drop *_genParticles_*_*',
      'drop *_genParticlesForJets_*_*',
      'drop *_kt4GenJets_*_*',
      'drop *_kt6GenJets_*_*',
      'drop *_iterativeCone5GenJets_*_*',
      'drop *_ak4GenJets_*_*',
      'drop *_ak7GenJets_*_*',
      'drop *_ak8GenJets_*_*',
      'drop *_ak4GenJetsNoNu_*_*',
      'drop *_ak8GenJetsNoNu_*_*',
      'drop *_genCandidatesForMET_*_*',
      'drop *_genParticlesForMETAllVisible_*_*',
      'drop *_genMetCalo_*_*',
      'drop *_genMetCaloAndNonPrompt_*_*',
      'drop *_genMetTrue_*_*',
      'drop *_genMetIC5GenJs_*_*'
      )
)

def SwapKeepAndDrop(l):
    r=[]
    for item in l:
        if 'keep ' in item:
            r.append(item.replace('keep ','drop '))
        elif 'drop ' in item:
            r.append(item.replace('drop ','keep '))
    return r

RESIMEventContent = cms.PSet(
    inputCommands=cms.untracked.vstring('drop *')
    )
RESIMEventContent.inputCommands.extend(IOMCRAW.outputCommands)
RESIMEventContent.inputCommands.extend(GeneratorInterfaceRAW.outputCommands)
#RESIMEventContent.inputCommands.extend(SwapKeepAndDrop(SimG4CoreRAW.outputCommands))
#RESIMEventContent.inputCommands.extend(SwapKeepAndDrop(GeneratorInterfaceRAW.outputCommands))

REDIGIEventContent = cms.PSet(
    inputCommands=cms.untracked.vstring('drop *')
    )
REDIGIEventContent.inputCommands.extend(SimG4CoreRAW.outputCommands)
REDIGIEventContent.inputCommands.extend(IOMCRAW.outputCommands)
REDIGIEventContent.inputCommands.extend(GeneratorInterfaceRAW.outputCommands)
REDIGIEventContent.inputCommands.append('drop *_randomEngineStateProducer_*_*')


########### and mini AOD
#
# MiniAOD is a bit special: the files tend to be so small that letting
# ROOT automatically determine when to flush is a surprisingly big overhead.
#

MINIAODEventContent= cms.PSet(    
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(-900),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)

MINIAODSIMEventContent= cms.PSet(    
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(-900),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)

MINIGENEventContent= cms.PSet(    
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(15*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)

from PhysicsTools.PatAlgos.slimming.slimming_cff import MicroEventContent,MicroEventContentMC,MicroEventContentGEN

MINIAODEventContent.outputCommands.extend(MicroEventContent.outputCommands)
MINIAODSIMEventContent.outputCommands.extend(MicroEventContentMC.outputCommands)
MINIGENEventContent.outputCommands.extend(MicroEventContentGEN.outputCommands)

#### RAW+miniAOD

RAWMINIAODEventContent= cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(20*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)

RAWMINIAODSIMEventContent= cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(20*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)

RAWMINIAODEventContent.outputCommands.extend(MicroEventContent.outputCommands)
RAWMINIAODEventContent.outputCommands.extend(L1TriggerRAW.outputCommands)
RAWMINIAODEventContent.outputCommands.extend(HLTriggerRAW.outputCommands)
RAWMINIAODSIMEventContent.outputCommands.extend(MicroEventContentMC.outputCommands)
RAWMINIAODSIMEventContent.outputCommands.extend(SimG4CoreHLTAODSIM.outputCommands)

RAWMINIAODSIMEventContent.outputCommands.extend(L1TriggerRAW.outputCommands)
RAWMINIAODSIMEventContent.outputCommands.extend(HLTriggerRAW.outputCommands)
RAWMINIAODEventContent.outputCommands.extend(cms.untracked.vstring(
    'keep FEDRawDataCollection_rawDataCollector_*_*',
    'keep FEDRawDataCollection_source_*_*'
))
RAWMINIAODSIMEventContent.outputCommands.extend(cms.untracked.vstring(
    'keep FEDRawDataCollection_rawDataCollector_*_*',
    'keep FEDRawDataCollection_source_*_*'
))


#
#
# RAWSIM Data Tier definition
# Meant as means to temporarily hold the RAW + AODSIM information as to allow the
# L1+HLT to be rerun at a later time.
#
RAWAODSIMEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *'),
    eventAutoFlushCompressedSize=cms.untracked.int32(20*1024*1024),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(4)
)

RAWAODSIMEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
RAWAODSIMEventContent.outputCommands.extend(L1TriggerRAW.outputCommands)
RAWAODSIMEventContent.outputCommands.extend(HLTriggerRAW.outputCommands)
RAWAODSIMEventContent.outputCommands.extend(SimG4CoreHLTAODSIM.outputCommands)

# in fastsim, normal digis are edaliases of simdigis
# drop the simdigis to avoid complaints from the outputmodule related to duplicated branches
for _entry in [FEVTDEBUGHLTEventContent,FEVTDEBUGEventContent,RECOSIMEventContent,AODSIMEventContent,RAWAODSIMEventContent]:
    fastSim.toModify(_entry, outputCommands = _entry.outputCommands + fastSimEC.dropSimDigis)
for _entry in [MINIAODEventContent, MINIAODSIMEventContent]:
    fastSim.toModify(_entry, outputCommands = _entry.outputCommands + fastSimEC.dropPatTrigger)


for _entry in [FEVTDEBUGEventContent,FEVTDEBUGHLTEventContent,FEVTEventContent]:
    phase2_tracker.toModify(_entry, outputCommands = _entry.outputCommands + [
        'keep Phase2TrackerDigiedmDetSetVector_mix_*_*',
        'keep *_TTClustersFromPhase2TrackerDigis_*_*',
        'keep *_TTStubsFromPhase2TrackerDigis_*_*'
    ])

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
for _entry in [FEVTDEBUGEventContent,FEVTDEBUGHLTEventContent,FEVTEventContent]:
    run2_GEM_2017.toModify(_entry, outputCommands = _entry.outputCommands + ['keep *_muonGEMDigis_*_*'])
    run3_GEM.toModify(_entry, outputCommands = _entry.outputCommands + ['keep *_muonGEMDigis_*_*'])
    phase2_muon.toModify(_entry, outputCommands = _entry.outputCommands + ['keep *_muonGEMDigis_*_*'])
    pp_on_AA_2018.toModify(_entry, outputCommands = _entry.outputCommands + ['keep FEDRawDataCollection_rawDataRepacker_*_*'])

from RecoLocalFastTime.Configuration.RecoLocalFastTime_EventContent_cff import RecoLocalFastTimeFEVT, RecoLocalFastTimeRECO, RecoLocalFastTimeAOD
from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
def _addOutputCommands(mod, newCommands):
    phase2_timing_layer.toModify(mod, outputCommands = mod.outputCommands + newCommands.outputCommands)

_addOutputCommands(FEVTDEBUGEventContent,RecoLocalFastTimeFEVT)
_addOutputCommands(FEVTDEBUGHLTEventContent,RecoLocalFastTimeFEVT)
_addOutputCommands(FEVTEventContent,RecoLocalFastTimeFEVT)
_addOutputCommands(RECOSIMEventContent,RecoLocalFastTimeRECO)
_addOutputCommands(AODSIMEventContent,RecoLocalFastTimeAOD)

from RecoMTD.Configuration.RecoMTD_EventContent_cff import RecoMTDFEVT, RecoMTDRECO, RecoMTDAOD
_addOutputCommands(FEVTDEBUGEventContent,RecoMTDFEVT)
_addOutputCommands(FEVTDEBUGHLTEventContent,RecoMTDFEVT)
_addOutputCommands(FEVTEventContent,RecoMTDFEVT)
_addOutputCommands(RECOSIMEventContent,RecoMTDRECO)
_addOutputCommands(AODSIMEventContent,RecoMTDAOD)
