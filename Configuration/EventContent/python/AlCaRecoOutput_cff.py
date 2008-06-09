import FWCore.ParameterSet.Config as cms

# last update: $Date: 2008/04/07 21:26:40 $ by $Author: futyand $
# Tracker Alignment
# AlCaReco for track based alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_Output_cff import *
# AlCaReco for track based alignment using Cosmic muon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_Output_cff import *
# AlCaReco for track based alignment using isoMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolated_Output_cff import *
# AlCaReco for track based alignment using J/Psi events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMu_Output_cff import *
# AlCaReco for track based alignment using Upsilon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_Output_cff import *
# AlCaReco for track based alignment using MinBias events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_Output_cff import *
# AlCaReco for pixel calibration using muons
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelLorentzAngle_Output_cff import *
# AlCaReco for tracker calibration using MinBias events
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBias_Output_cff import *
# ECAL Calibration
# ECAL calibration with phi symmetry 
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPhiSym_Output_cff import *
# ECAL calibration with isol. electrons
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalElectron_Output_cff import *
# The following paths are obsoleted since pi0 calibration
# has a HLT path (argiro,20080314 )
# ECAL calibration with pi0 
#  include "Calibration/EcalAlCaRecoProducers/data/ALCARECOEcalCalPi0_Output.cff"
# ECAL calibration with pi0 Basic Clusters
#  include "Calibration/EcalAlCaRecoProducers/data/ALCARECOEcalCalPi0BC_Output.cff"
# ECAL calibration with pi0 hlt path
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalPi0Calib_Output_cff import *
# HCAL Calibration
# HCAL calibration with dijets
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalDijets_Output_cff import *
# HCAL calibration with gamma + jet
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_Output_cff import *
# HCAL calibration with isolated tracks
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_Output_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkNoHLT_Output_cff import *
# HCAl calibration with min.bias
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalMinBias_Output_cff import *
# HCAl calibration with Zmuu (HO)
#  include "Calibration/HcalAlCaRecoProducers/data/ALCARECOHcalCalZMuMu_Output.cff"
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_Output_cff import *
# Muon Alignment with Zmumu
from Alignment.CommonAlignmentProducer.ALCARECOMuAlZMuMu_Output_cff import *
# Muon Alignment using CSC overlaps
from Alignment.CommonAlignmentProducer.ALCARECOMuAlOverlaps_Output_cff import *
# RPC calibration
from CalibMuon.RPCCalibration.ALCARECORpcCalHLT_Output_cff import *

