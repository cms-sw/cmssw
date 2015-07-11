import FWCore.ParameterSet.Config as cms
# last update: $Date: 2012/03/30 17:07:33 $ by $Author: cerminar $
###############################################################
# Tracker Alignment
###############################################################
# AlCaReco for track based alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_Output_cff import *
# AlCaReco for track based alignment using Cosmic muon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsInCollisions_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsHLT_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0THLT_Output_cff import *
# AlCaReco for track based alignment using Laser events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlLAS_Output_cff import *
# AlCaReco for track based alignment using isoMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolated_Output_cff import *
# AlCaReco for track based alignment using isoMu events for PA data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolatedPA_Output_cff import *
# AlCaReco for track based alignment using J/Psi events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMu_Output_cff import *
# AlCaReco for track based alignment using Upsilon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_Output_cff import *
# AlCaReco for track based alignment using MinBias events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBiasHI_Output_cff import *


# AlCaReco for pixel calibration using muons
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelLorentzAngle_Output_cff import *
# AlCaReco for tracker calibration using MinBias events
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBias_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalZeroBias_Output_cff import *

# AlCaReco for tracker based alignment using beam halo 
from Alignment.CommonAlignmentProducer.ALCARECOTkAlBeamHalo_Output_cff import *
###############################################################
# ECAL Calibration
###############################################################
# ECAL calibration with isol. electrons
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_Output_cff import *
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalUncalIsolElectron_Output_cff import *
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalRecalIsolElectron_Output_cff import *

# The following paths are obsoleted since pi0 calibration
# has a HLT path (argiro,20080314 )
# ECAL calibration with pi0 
#  include "Calibration/EcalAlCaRecoProducers/data/ALCARECOEcalCalPi0_Output.cff"
# ECAL calibration with pi0 Basic Clusters
#  include "Calibration/EcalAlCaRecoProducers/data/ALCARECOEcalCalPi0BC_Output.cff"
# ECAL calibration with pi0 hlt path
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPi0Calib_Output_cff import *
# ECAL calibration with eta hlt path
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalEtaCalib_Output_cff import *
###############################################################
# HCAL Calibration
###############################################################
# HCAL calibration with dijets
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalDijets_Output_cff import *
# HCAL calibration with gamma + jet
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_Output_cff import *
# HCAL calibration with isolated tracks
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_Output_cff import *
# HCAL calibration with iterative phi sym                                                                                       
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIterativePhiSym_Output_cff import *
# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBiasHI_Output_cff import *
# HCAL calibration with Zmuu (HO)
#  include "Calibration/HcalAlCaRecoProducers/data/ALCARECOHcalCalZMuMu_Output.cff"
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHOCosmics_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalNoise_Output_cff import *
###############################################################
# Muon Alignment (incl. stream for calibration)
###############################################################
# Muon Alignment with cosmics
from Alignment.CommonAlignmentProducer.ALCARECOMuAlStandAloneCosmics_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlGlobalCosmicsInCollisions_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlGlobalCosmics_Output_cff import *
# Muon Alignment with isolated muons
from Alignment.CommonAlignmentProducer.ALCARECOMuAlCalIsolatedMu_Output_cff import *
# Muon Alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOMuAlZMuMu_Output_cff import *
# Muon Alignment using CSC overlaps
from Alignment.CommonAlignmentProducer.ALCARECOMuAlOverlaps_Output_cff import *
# Muon Alignment using beam halo
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHaloOverlaps_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHalo_Output_cff import *
###############################################################
# RPC calibration
###############################################################
from CalibMuon.RPCCalibration.ALCARECORpcCalHLT_Output_cff import *
###############################################################
# DT calibration
###############################################################
from CalibMuon.DTCalibration.ALCARECODtCalib_Output_cff import *
from CalibMuon.DTCalibration.ALCARECODtCalibHI_Output_cff import *
from CalibMuon.DTCalibration.ALCARECODtCalibCosmics_Output_cff import *

###############################################################
# stream for prompt-calibration @ Tier0
###############################################################
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProd_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStrip_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGains_Output_cff import *

from Calibration.TkAlCaRecoProducers.ALCARECOSiStripPCLHistos_Output_cff import *

# stream for the LumiPixels workflow
from Calibration.TkAlCaRecoProducers.ALCARECOLumiPixels_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOLumiPixelsMinBias_Output_cff import *

###############################################################
# hotline skim workflows
###############################################################
from Calibration.Hotline.hotlineSkims_Output_cff import *

ALCARECOEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *',
        'keep edmTriggerResults_*_*_*'),
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize=cms.untracked.int32(5*1024*1024)
)

ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlZMuMu_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlCosmicsInCollisions_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlCosmics_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlCosmicsHLT_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlCosmics0T_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlCosmics0THLT_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlLAS_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlMuonIsolated_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlMuonIsolatedPA_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlJpsiMuMu_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlUpsilonMuMu_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalIterativePhiSym_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlMinBias_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlBeamHalo_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOSiStripCalZeroBias_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOSiStripCalMinBias_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOEcalCalElectron_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOEcalUncalElectron_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOEcalCalPi0Calib_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOEcalCalEtaCalib_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalDijets_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalGammaJet_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalIsoTrk_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalIsoTrkNoHLT_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalMinBias_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalHO_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalHOCosmics_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalNoise_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlStandAloneCosmics_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlGlobalCosmicsInCollisions_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlGlobalCosmics_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlCalIsolatedMu_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlZMuMu_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlOverlaps_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlBeamHaloOverlaps_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOMuAlBeamHalo_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECORpcCalHLT_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECODtCalib_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOSiStripPCLHistos_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOLumiPixels_noDrop.outputCommands)
#ALCARECOEventContent.outputCommands.extend(OutALCARECOHotline.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOTkAlMinBiasHI_noDrop.outputCommands)
ALCARECOEventContent.outputCommands.extend(OutALCARECOHcalCalMinBiasHI_noDrop.outputCommands)


ALCARECOEventContent.outputCommands.append('drop *_MEtoEDMConverter_*_*')
