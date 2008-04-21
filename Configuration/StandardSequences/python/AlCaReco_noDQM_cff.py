import FWCore.ParameterSet.Config as cms

# last update: $Date: 2008/04/17 14:58:03 $ by $Author: futyand $
# Please update the sequence defined at the very end of this file
# with any new/changed sequences
# Tracker Alignment
# AlCaReco for track based alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_cff import *
# AlCaReco for track based alignment using Cosmic muon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *
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
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_cff import *
# HCAl calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_cff import *
# Hcal calibration from HO (muons) 
#  include "Calibration/HcalAlCaRecoProducers/data/ALCARECOHcalCalZMuMu.cff"
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_cff import *
# Muon Alignment with Zmumu
from Alignment.CommonAlignmentProducer.ALCARECOMuAlZMuMu_cff import *
# Muon Alignment using CSC overlaps
from Alignment.CommonAlignmentProducer.ALCARECOMuAlOverlaps_cff import *
# RPC calibration
from CalibMuon.RPCCalibration.ALCARECORpcCalHLT_cff import *
pathALCARECOTkAlZMuMu = cms.Path(seqALCARECOTkAlZMuMu)
pathALCARECOTkAlMuonIsolated = cms.Path(seqALCARECOTkAlMuonIsolated)
pathALCARECOTkAlJpsiMuMu = cms.Path(seqALCARECOTkAlJpsiMuMu)
pathALCARECOTkAlUpsilonMuMu = cms.Path(seqALCARECOTkAlUpsilonMuMu)
pathALCARECOTkAlMinBias = cms.Path(seqALCARECOTkAlMinBias)
pathALCARECOSiPixelLorentzAngle = cms.Path(seqALCARECOSiPixelLorentzAngle)
pathALCARECOSiStripCalMinBias = cms.Path(seqALCARECOSiStripCalMinBias)
pathALCARECOEcalCalElectron = cms.Path(seqALCARECOEcalCalElectron)
pathALCARECOEcalCalPhiSym = cms.Path(seqALCARECOEcalCalPhiSym)
pathALCARECOEcalCalPi0Calib = cms.Path(seqALCARECOEcalCalPi0Calib)
pathALCARECOHcalCalMinBias = cms.Path(seqALCARECOHcalCalMinBias)
pathALCARECOHcalCalDijets = cms.Path(seqALCARECOHcalCalDijets)
pathALCARECOHcalCalGammaJet = cms.Path(seqALCARECOHcalCalGammaJet)
pathALCARECOHcalCalIsoTrkNoHLT = cms.Path(seqALCARECOHcalCalIsoTrkNoHLT)
pathALCARECOHcalCalHO = cms.Path(seqALCARECOHcalCalHO)
pathALCARECOMuAlZMuMu = cms.Path(seqALCARECOMuAlZMuMu)
pathALCARECOMuAlOverlaps = cms.Path(seqALCARECOMuAlOverlaps)
pathALCARECORpcCalHLT = cms.Path(seqALCARECORpcCalHLT)

