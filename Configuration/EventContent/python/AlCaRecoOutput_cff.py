import FWCore.ParameterSet.Config as cms
# last update: $Date: 2012/03/30 17:07:33 $ by $Author: cerminar $
###############################################################
# Tracker Alignment
###############################################################
# AlCaReco for track based alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_Output_cff import *
# AlCaReco for track based alignment using ZMuMu events for PbPb data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMuHI_Output_cff import *
# AlCaReco for track based alignment using ZMuMu events for PA data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMuPA_Output_cff import *
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
# AlCaReco for track based alignment using isoMu events for PbPb data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolatedHI_Output_cff import *
# AlCaReco for track based alignment using isoMu events for PA data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolatedPA_Output_cff import *
# AlCaReco for track based alignment using J/Psi events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMu_Output_cff import *
# AlCaReco for track based alignment using J/Psi events for PbPb data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMuHI_Output_cff import *
# AlCaReco for track based alignment using Upsilon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_Output_cff import *
# AlCaReco for track based alignment using Upsilon events for PbPb data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMuHI_Output_cff import *
# AlCaReco for track based alignment using Upsilon events for PA data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMuPA_Output_cff import *
# AlCaReco for track based alignment using MinBias events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_Output_cff import *
# AlCaReco for track based alignment using MinBias events for PbPb data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBiasHI_Output_cff import *

# AlCaReco for pixel calibration using muons
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelLorentzAngle_Output_cff import *
# AlCaReco for tracker calibration using MinBias events
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBias_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBiasAAG_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalSmallBiasScan_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalZeroBias_Output_cff import *
# AlCaReco for SiPixel Bad Component using ZeroBias events
from CalibTracker.SiPixelQuality.ALCARECOSiPixelCalZeroBias_Output_cff import *

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
# ECAL ES alignment
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalESAlign_Output_cff import *
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalTrg_Output_cff import *
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalTestPulsesRaw_Output_cff import *

###############################################################
# HCAL Calibration
###############################################################
# HCAL calibration with dijets
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalDijets_Output_cff import *
# HCAL calibration with gamma + jet
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_Output_cff import *
# HCAL calibration with isolated tracks
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkFilter_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_Output_cff import *
# HCAL calibration with iterative phi sym
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIterativePhiSym_Output_cff import *
# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBiasHI_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalPedestal_Output_cff import *
# HCAL calibration with Zmuu (HO)
#  include "Calibration/HcalAlCaRecoProducers/data/ALCARECOHcalCalZMuMu_Output.cff"
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHOCosmics_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalNoise_Output_cff import *
# HCAL calibration with isolated bunch
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsolatedBunchFilter_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsolatedBunchSelector_Output_cff import *
# HCAL calibration with muons (HB/HE)
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHBHEMuonFilter_Output_cff import *
# HCAL calibration with muons (HE)
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHEMuonFilter_Output_cff import *
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
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdBeamSpotHP_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdBeamSpotHPLowPU_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStrip_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGains_Output_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGainsAAG_Output_cff import *

from Calibration.TkAlCaRecoProducers.ALCARECOSiStripPCLHistos_Output_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAli_Output_cff import *

from CalibTracker.SiPixelQuality.ALCARECOPromptCalibProdSiPixel_Output_cff import *

from Calibration.EcalCalibAlgos.ALCARECOPromptCalibProdEcalPedestals_Output_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECOPromptCalibProdLumiPCC_Output_cff import *
# Pixel Cluster Counting ALCARECOs
# in AlCaLumiPixels stream
from Calibration.LumiAlCaRecoProducers.ALCARECOLumiPixels_Output_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECOAlCaPCCZeroBias_Output_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECOAlCaPCCRandom_Output_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECORawPCCProducer_Output_cff import *

# on top of prompt RECO
from Calibration.LumiAlCaRecoProducers.ALCARECOLumiPixelsMinBias_Output_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECOAlCaPCCZeroBiasFromRECO_Output_cff import *
from Calibration.LumiAlCaRecoProducers.ALCARECOAlCaPCCRandomFromRECO_Output_cff import *

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



ALCARECOEventContent.outputCommands.append('drop *_MEtoEDMConverter_*_*')
