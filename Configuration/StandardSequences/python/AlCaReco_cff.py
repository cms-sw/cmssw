import FWCore.ParameterSet.Config as cms

# last update: $Date: 2008/08/14 08:28:12 $ by $Author: futyand $
# Please update the sequence defined at the very end of this file
# with any new/changed sequences
# Tracker Alignment
# AlCaReco for track based alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_cff import *
# AlCaReco for track based alignment using Cosmic muon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsHLT_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0THLT_cff import *
# AlCaReco for track based alignment using isoMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolated_cff import *
# AlCaReco for track based alignment using J/Psi events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMu_cff import *
# AlCaReco for track based alignment using Upsilon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_cff import *
# AlCaReco for track based alignment using MinBias events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import *
# AlCaReco for pixel calibration using muons
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelLorentzAngle_cff import *
# AlCaReco for tracker calibration using MinBias events
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBias_cff import *
# ECAL Calibration
# ECAL calibration with phi symmetry 
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPhiSym_cff import *
# ECAL calibration with isol. electrons
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalElectron_cff import *
# The following paths are obsoleted since pi0 calibration
# has a HLT path (argiro,20080314 )
# ECAL calibration with pi0
#  include "Calibration/EcalAlCaRecoProducers/data/ALCARECOEcalCalPi0.cff"
# ECAL calibration with pi0 Basic Clusters
#  include "Calibration/EcalAlCaRecoProducers/data/ALCARECOEcalCalPi0BC.cff"
# ECAL calibration with pi0
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPi0Calib_cff import *
# HCAL Calibration
# HCAL calibration with dijets
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalDijets_cff import *
# HCAL calibration with gamma+jet
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_cff import *
# HCAL calibration with isolated tracks
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff import *
# HCAL calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *
# HCAL calibration from HO (muons) 
#  include "Calibration/HcalAlCaRecoProducers/data/ALCARECOHcalCalZMuMu.cff"
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_cff import *
# Muon Alignment with cosmics
from Alignment.CommonAlignmentProducer.ALCARECOMuAlStandAloneCosmics_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlGlobalCosmics_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlZeroFieldGlobalCosmics_cff import *
# Muon calibration with minbias
from Alignment.CommonAlignmentProducer.ALCARECOMuCaliMinBias_cff import *
# Muon Alignment with isolated muons
from Alignment.CommonAlignmentProducer.ALCARECOMuAlCalIsolatedMu_cff import *
# Muon Alignment using CSC overlaps
from Alignment.CommonAlignmentProducer.ALCARECOMuAlOverlaps_cff import *
# RPC calibration
from CalibMuon.RPCCalibration.ALCARECORpcCalHLT_cff import *
# nonbeam alcas
from Alignment.CommonAlignmentProducer.ALCARECOTkAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlLAS_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHaloOverlaps_cff import *
# NOTE: the ALCARECO DQM modules can not be placed together in a single path 
# because they act as filters. They are therefore inserted per ALCARECO path.
from DQMOffline.Configuration.AlCaRecoDQM_cff import *
pathALCARECOTkAlZMuMu = cms.Path(seqALCARECOTkAlZMuMu*ALCARECOTkAlZMuMuDQM)
pathALCARECOTkAlMuonIsolated = cms.Path(seqALCARECOTkAlMuonIsolated*ALCARECOTkAlMuonIsolatedDQM)
pathALCARECOTkAlJpsiMuMu = cms.Path(seqALCARECOTkAlJpsiMuMu*ALCARECOTkAlJpsiMuMuDQM)
pathALCARECOTkAlUpsilonMuMu = cms.Path(seqALCARECOTkAlUpsilonMuMu*ALCARECOTkAlUpsilonMuMuDQM)
pathALCARECOTkAlMinBias = cms.Path(seqALCARECOTkAlMinBias*ALCARECOTkAlMinBiasDQM)
pathALCARECOSiPixelLorentzAngle = cms.Path(seqALCARECOSiPixelLorentzAngle)
pathALCARECOSiStripCalMinBias = cms.Path(seqALCARECOSiStripCalMinBias)
pathALCARECOEcalCalElectron = cms.Path(seqALCARECOEcalCalElectron)
pathALCARECOEcalCalPhiSym = cms.Path(seqALCARECOEcalCalPhiSym*ALCARECOEcalCalPhisymDQM)
pathALCARECOEcalCalPi0Calib = cms.Path(seqALCARECOEcalCalPi0Calib*ALCARECOEcalCalPi0CalibDQM)
pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBias)
pathALCARECOHcalCalDijets = cms.Path(seqALCARECOHcalCalDijets)
pathALCARECOHcalCalGammaJet = cms.Path(seqALCARECOHcalCalGammaJet)
pathALCARECOHcalCalIsoTrk = cms.Path(seqALCARECOHcalCalIsoTrk)
pathALCARECOHcalCalHO = cms.Path(seqALCARECOHcalCalHO)
pathALCARECOMuCaliMinBias = cms.Path(seqALCARECOMuCaliMinBias)
pathALCARECOMuAlCalIsolatedMu = cms.Path(seqALCARECOMuAlCalIsolatedMu)
pathALCARECOMuAlOverlaps = cms.Path(seqALCARECOMuAlOverlaps)
pathALCARECORpcCalHLT = cms.Path(seqALCARECORpcCalHLT)
pathALCARECOTkAlBeamHalo = cms.Path(seqALCARECOTkAlBeamHalo)
pathALCARECOMuAlBeamHaloOverlaps = cms.Path(seqALCARECOMuAlBeamHaloOverlaps)
pathALCARECOMuAlBeamHalo = cms.Path(seqALCARECOMuAlBeamHalo)
pathALCARECOTkAlLAS = cms.Path(seqALCARECOTkAlLAS)
pathALCARECOTkAlCosmicsCTF = cms.Path(seqALCARECOTkAlCosmicsCTF)
pathALCARECOTkAlCosmicsCosmicTF = cms.Path(seqALCARECOTkAlCosmicsCosmicTF)
pathALCARECOTkAlCosmicsRS = cms.Path(seqALCARECOTkAlCosmicsRS)
pathALCARECOTkAlCosmicsCTF0T = cms.Path(seqALCARECOTkAlCosmicsCTF0T)
pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(seqALCARECOTkAlCosmicsCosmicTF0T)
pathALCARECOTkAlCosmicsRS0T = cms.Path(seqALCARECOTkAlCosmicsRS0T) 
pathALCARECOTkAlCosmicsCTFHLT = cms.Path(seqALCARECOTkAlCosmicsCTFHLT)
pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(seqALCARECOTkAlCosmicsCosmicTFHLT)
pathALCARECOTkAlCosmicsRSHLT = cms.Path(seqALCARECOTkAlCosmicsRSHLT)
pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(seqALCARECOTkAlCosmicsCTF0THLT)
pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(seqALCARECOTkAlCosmicsCosmicTF0THLT)
pathALCARECOTkAlCosmicsRS0THLT = cms.Path(seqALCARECOTkAlCosmicsRS0THLT)
pathALCARECOMuAlStandAloneCosmics = cms.Path(seqALCARECOMuAlStandAloneCosmics)
pathALCARECOMuAlGlobalCosmics = cms.Path(seqALCARECOMuAlGlobalCosmics)
pathALCARECOMuAlZeroFieldGlobalCosmics = cms.Path(seqALCARECOMuAlZeroFieldGlobalCosmics)

