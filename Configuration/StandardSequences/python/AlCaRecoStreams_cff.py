import FWCore.ParameterSet.Config as cms

# last update: $Date: 2013/02/05 10:02:31 $ by $Author: cerminar $

# AlCaReco sequence definitions:

###############################################################
# Tracker Alignment
###############################################################
# AlCaReco for track based alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMu_cff import *
# AlCaReco for track based alignment using Cosmic muon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsInCollisions_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsHLT_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0THLT_cff import *
# AlCaReco for track based alignment using isoMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolated_cff import *
# AlCaReco for track based alignment using isoMu events for PA data-taking
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolatedPA_cff import *
# AlCaReco for track based alignment using J/Psi events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMu_cff import *
# AlCaReco for track based alignment using Upsilon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMu_cff import *
# AlCaReco for track based alignment using MinBias events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBias_cff import *

###############################################################
# Tracker Calibration
###############################################################
# AlCaReco for pixel calibration using muons 
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelLorentzAngle_cff import *
# AlCaReco for tracker calibration using MinBias events
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBias_cff import *
# AlCaReco for tracker calibration using ZeroBias events (noise)
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalZeroBias_cff import *

###############################################################
# ECAL Calibration
###############################################################
# ECAL calibration with isol. electrons
from Calibration.EcalAlCaRecoProducers.ALCARECOEcalCalIsolElectron_cff import *
###############################################################
# HCAL Calibration
###############################################################
# HCAL calibration with dijets
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalDijets_cff import *
# HCAL calibration with gamma+jet (obsolete)
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_cff import *
# HCAL calibration from HO (muons) 
#  include "Calibration/HcalAlCaRecoProducers/data/ALCARECOHcalCalZMuMu.cff"
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHOCosmics_cff import *
# HCAL isotrack
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrk_cff import *
# HCAL noise
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalNoise_cff import *

###############################################################
# Muon alignment
###############################################################
# Muon Alignment with cosmics
from Alignment.CommonAlignmentProducer.ALCARECOMuAlGlobalCosmicsInCollisions_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlGlobalCosmics_cff import *
# Muon Alignment/Calibration with isolated muons
from Alignment.CommonAlignmentProducer.ALCARECOMuAlCalIsolatedMu_cff import *
# Muon Alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOMuAlZMuMu_cff import *
# Muon Alignment using CSC overlaps
from Alignment.CommonAlignmentProducer.ALCARECOMuAlOverlaps_cff import *
###############################################################
# RPC calibration
###############################################################
from CalibMuon.RPCCalibration.ALCARECORpcCalHLT_cff import *
###############################################################
# DT calibration
###############################################################
from CalibMuon.DTCalibration.ALCARECODtCalib_cff import *
from CalibMuon.DTCalibration.ALCARECODtCalibCosmics_cff import *


###############################################################
# nonbeam alcas
###############################################################
from Alignment.CommonAlignmentProducer.ALCARECOTkAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlLAS_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHaloOverlaps_cff import *

###############################################################
# stream for prompt-calibration @ Tier0
###############################################################
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProd_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripPCLHistos_cff import *




# NOTE: the ALCARECO DQM modules can not be placed together in a single path 
# because the non-DQM sequences act as filters.
# They are therefore inserted per ALCARECO path.
from DQMOffline.Configuration.AlCaRecoDQM_cff import *

# AlCaReco path definitions:

pathALCARECOTkAlZMuMu = cms.Path(seqALCARECOTkAlZMuMu*ALCARECOTkAlZMuMuDQM)
pathALCARECOTkAlMuonIsolated = cms.Path(seqALCARECOTkAlMuonIsolated*ALCARECOTkAlMuonIsolatedDQM)
pathALCARECOTkAlMuonIsolatedPA = cms.Path(seqALCARECOTkAlMuonIsolatedPA)
pathALCARECOTkAlJpsiMuMu = cms.Path(seqALCARECOTkAlJpsiMuMu*ALCARECOTkAlJpsiMuMuDQM)
pathALCARECOTkAlUpsilonMuMu = cms.Path(seqALCARECOTkAlUpsilonMuMu*ALCARECOTkAlUpsilonMuMuDQM)
pathALCARECOTkAlMinBias = cms.Path(seqALCARECOTkAlMinBias*ALCARECOTkAlMinBiasDQM)
pathALCARECOTkAlMinBias = cms.Path(seqALCARECOTkAlMinBias*ALCARECOTkAlMinBiasDQM)
pathALCARECOSiPixelLorentzAngle = cms.Path(seqALCARECOSiPixelLorentzAngle)
pathALCARECOSiStripCalMinBias = cms.Path(seqALCARECOSiStripCalMinBias*ALCARECOSiStripCalMinBiasDQM)
pathALCARECOSiStripCalZeroBias = cms.Path(seqALCARECOSiStripCalZeroBias*ALCARECOSiStripCalZeroBiasDQM)

pathALCARECOEcalCalZElectron = cms.Path(seqALCARECOEcalCalZElectron)
pathALCARECOEcalCalWElectron = cms.Path(seqALCARECOEcalCalWElectron)

pathALCARECOHcalCalDijets = cms.Path(seqALCARECOHcalCalDijets*ALCARECOHcalCalDiJetsDQM)
pathALCARECOHcalCalGammaJet = cms.Path(seqALCARECOHcalCalGammaJet)
pathALCARECOHcalCalHO = cms.Path(seqALCARECOHcalCalHO*ALCARECOHcalCalHODQM)
pathALCARECOHcalCalHOCosmics = cms.Path(seqALCARECOHcalCalHOCosmics)
pathALCARECOHcalCalIsoTrk = cms.Path(seqALCARECOHcalCalIsoTrk*ALCARECOHcalCalIsoTrackDQM)
pathALCARECOHcalCalNoise = cms.Path(seqALCARECOHcalCalNoise)
pathALCARECOMuAlCalIsolatedMu = cms.Path(seqALCARECOMuAlCalIsolatedMu*ALCARECOMuAlCalIsolatedMuDQM*ALCARECODTCalibrationDQM)
pathALCARECOMuAlZMuMu = cms.Path(seqALCARECOMuAlZMuMu*ALCARECOMuAlZMuMuDQM)
pathALCARECOMuAlOverlaps = cms.Path(seqALCARECOMuAlOverlaps*ALCARECOMuAlOverlapsDQM)
pathALCARECORpcCalHLT = cms.Path(seqALCARECORpcCalHLT)
pathALCARECODtCalib = cms.Path(seqALCARECODtCalib*ALCARECODTCalibSynchDQM)
pathALCARECODtCalibCosmics = cms.Path(seqALCARECODtCalibCosmics*ALCARECODTCalibSynchCosmicsDQM)
pathALCARECOTkAlBeamHalo = cms.Path(seqALCARECOTkAlBeamHalo*ALCARECOTkAlBeamHaloDQM)
pathALCARECOMuAlBeamHaloOverlaps = cms.Path(seqALCARECOMuAlBeamHaloOverlaps*ALCARECOMuAlBeamHaloOverlapsDQM)
pathALCARECOMuAlBeamHalo = cms.Path(seqALCARECOMuAlBeamHalo*ALCARECOMuAlBeamHaloDQM)
pathALCARECOTkAlLAS = cms.Path(seqALCARECOTkAlLAS*ALCARECOTkAlLASDQM)
pathALCARECOTkAlCosmicsInCollisions = cms.Path(seqALCARECOTkAlCosmicsInCollisions*ALCARECOTkAlCosmicsInCollisionsDQM)
pathALCARECOTkAlCosmicsCTF = cms.Path(seqALCARECOTkAlCosmicsCTF*ALCARECOTkAlCosmicsCTFDQM)
pathALCARECOTkAlCosmicsCosmicTF = cms.Path(seqALCARECOTkAlCosmicsCosmicTF*ALCARECOTkAlCosmicsCosmicTFDQM)
pathALCARECOTkAlCosmicsRegional = cms.Path(seqALCARECOTkAlCosmicsRegional*ALCARECOTkAlCosmicsRegionalDQM)
pathALCARECOTkAlCosmicsCTF0T = cms.Path(seqALCARECOTkAlCosmicsCTF0T*ALCARECOTkAlCosmicsCTF0TDQM)
pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(seqALCARECOTkAlCosmicsCosmicTF0T*ALCARECOTkAlCosmicsCosmicTF0TDQM)
pathALCARECOTkAlCosmicsRegional0T = cms.Path(seqALCARECOTkAlCosmicsRegional0T*ALCARECOTkAlCosmicsRegional0TDQM)
pathALCARECOTkAlCosmicsCTFHLT = cms.Path(seqALCARECOTkAlCosmicsCTFHLT*ALCARECOTkAlCosmicsCTFDQM)
pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(seqALCARECOTkAlCosmicsCosmicTFHLT*ALCARECOTkAlCosmicsCosmicTFDQM)
pathALCARECOTkAlCosmicsRegionalHLT = cms.Path(seqALCARECOTkAlCosmicsRegionalHLT*ALCARECOTkAlCosmicsRegionalDQM)
pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(seqALCARECOTkAlCosmicsCTF0THLT*ALCARECOTkAlCosmicsCTF0TDQM)
pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(seqALCARECOTkAlCosmicsCosmicTF0THLT*ALCARECOTkAlCosmicsCosmicTF0TDQM)
pathALCARECOTkAlCosmicsRegional0THLT = cms.Path(seqALCARECOTkAlCosmicsRegional0THLT*ALCARECOTkAlCosmicsRegional0TDQM)
pathALCARECOMuAlGlobalCosmicsInCollisions = cms.Path(seqALCARECOMuAlGlobalCosmicsInCollisions*ALCARECOMuAlGlobalCosmicsInCollisionsDQM)
pathALCARECOMuAlGlobalCosmics = cms.Path(seqALCARECOMuAlGlobalCosmics*ALCARECOMuAlGlobalCosmicsDQM)
pathALCARECOPromptCalibProd = cms.Path(seqALCARECOPromptCalibProd)
pathALCARECOSiStripPCLHistos = cms.Path(seqALCARECOSiStripPCLHistos)

# AlCaReco event content definitions:

from Configuration.EventContent.AlCaRecoOutput_cff import *

# AlCaReco stream definitions:

ALCARECOStreamTkAlMinBias = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlMinBias',
	paths  = (pathALCARECOTkAlMinBias),
	content = OutALCARECOTkAlMinBias.outputCommands,
	selectEvents = OutALCARECOTkAlMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlMuonIsolated = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlMuonIsolated',
	paths  = (pathALCARECOTkAlMuonIsolated),
	content = OutALCARECOTkAlMuonIsolated.outputCommands,
	selectEvents = OutALCARECOTkAlMuonIsolated.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlMuonIsolatedPA = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'TkAlMuonIsolatedPA',
	paths  = (pathALCARECOTkAlMuonIsolatedPA),
	content = OutALCARECOTkAlMuonIsolatedPA.outputCommands,
	selectEvents = OutALCARECOTkAlMuonIsolatedPA.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlZMuMu = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlZMuMu',
	paths  = (pathALCARECOTkAlZMuMu),
	content = OutALCARECOTkAlZMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlZMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlJpsiMuMu = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlJpsiMuMu',
	paths  = (pathALCARECOTkAlJpsiMuMu),
	content = OutALCARECOTkAlJpsiMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlJpsiMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlUpsilonMuMu = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlUpsilonMuMu',
	paths  = (pathALCARECOTkAlUpsilonMuMu),
	content = OutALCARECOTkAlUpsilonMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlUpsilonMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiPixelLorentzAngle = cms.FilteredStream(
	responsible = 'Lotte Wilke',
	name = 'SiPixelLorentzAngle',
	paths  = (pathALCARECOSiPixelLorentzAngle),
	content = OutALCARECOSiPixelLorentzAngle.outputCommands,
	selectEvents = OutALCARECOSiPixelLorentzAngle.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiStripCalMinBias = cms.FilteredStream(
	responsible = 'Vitaliano Ciulli',
	name = 'SiStripCalMinBias',
	paths  = (pathALCARECOSiStripCalMinBias),
	content = OutALCARECOSiStripCalMinBias.outputCommands,
	selectEvents = OutALCARECOSiStripCalMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiStripCalZeroBias = cms.FilteredStream(
	responsible = 'Gordon Kaussen',
	name = 'SiStripCalZeroBias',
	paths  = (pathALCARECOSiStripCalZeroBias),
	content = OutALCARECOSiStripCalZeroBias.outputCommands,
	selectEvents = OutALCARECOSiStripCalZeroBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamEcalCalElectron = cms.FilteredStream(
	responsible = 'Shervin Nourbakhsh',
	name = 'EcalCalElectron',
	paths  = (pathALCARECOEcalCalZElectron, pathALCARECOEcalCalWElectron),
	content = OutALCARECOEcalCalElectron.outputCommands,
	selectEvents = OutALCARECOEcalCalElectron.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalDijets = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'HcalCalDijets',
	paths  = (pathALCARECOHcalCalDijets),
	content = OutALCARECOHcalCalDijets.outputCommands,
	selectEvents = OutALCARECOHcalCalDijets.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalGammaJet = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'HcalCalGammaJet',
	paths  = (pathALCARECOHcalCalGammaJet),
	content = OutALCARECOHcalCalGammaJet.outputCommands,
	selectEvents = OutALCARECOHcalCalGammaJet.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalHO = cms.FilteredStream(
	responsible = 'Gobinda Majumder',
	name = 'HcalCalHO',
	paths  = (pathALCARECOHcalCalHO),
	content = OutALCARECOHcalCalHO.outputCommands,
	selectEvents = OutALCARECOHcalCalHO.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalHOCosmics = cms.FilteredStream(
	responsible = 'Gobinda Majumder',
	name = 'HcalCalHOCosmics',
	paths  = (pathALCARECOHcalCalHOCosmics),
	content = OutALCARECOHcalCalHOCosmics.outputCommands,
	selectEvents = OutALCARECOHcalCalHOCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalIsoTrk = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'HcalCalIsoTrk',
	paths  = (pathALCARECOHcalCalIsoTrk),
	content = OutALCARECOHcalCalIsoTrk.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrk.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalNoise = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'HcalCalNoise',
	paths  = (pathALCARECOHcalCalNoise),
	content = OutALCARECOHcalCalNoise.outputCommands,
	selectEvents = OutALCARECOHcalCalNoise.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)


ALCARECOStreamMuAlCalIsolatedMu = cms.FilteredStream(
	responsible = 'Javier Fernandez',
	name = 'MuAlCalIsolatedMu',
	paths  = (pathALCARECOMuAlCalIsolatedMu),
	content = OutALCARECOMuAlCalIsolatedMu.outputCommands,
	selectEvents = OutALCARECOMuAlCalIsolatedMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlZMuMu = cms.FilteredStream(
	responsible = 'Javier Fernandez',
	name = 'MuAlZMuMu',
	paths  = (pathALCARECOMuAlZMuMu),
	content = OutALCARECOMuAlZMuMu.outputCommands,
	selectEvents = OutALCARECOMuAlZMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlOverlaps = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'MuAlOverlaps',
	paths  = (pathALCARECOMuAlOverlaps),
	content = OutALCARECOMuAlOverlaps.outputCommands,
	selectEvents = OutALCARECOMuAlOverlaps.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamRpcCalHLT = cms.FilteredStream(
	responsible = 'Marcello Maggi',
	name = 'RpcCalHLT',
	paths  = (pathALCARECORpcCalHLT),
	content = OutALCARECORpcCalHLT.outputCommands,
	selectEvents = OutALCARECORpcCalHLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamDtCalib = cms.FilteredStream(
	responsible = 'Mario Pelliccioni',
	name = 'DtCalib',
	paths  = (pathALCARECODtCalib),
	content = OutALCARECODtCalib.outputCommands,
	selectEvents = OutALCARECODtCalib.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamDtCalibCosmics = cms.FilteredStream(
	responsible = 'Antonio Vilela Pereira',
	name = 'DtCalibCosmics',
	paths  = (pathALCARECODtCalibCosmics),
	content = OutALCARECODtCalibCosmics.outputCommands,
	selectEvents = OutALCARECODtCalibCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmicsInCollisions = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlCosmicsInCollisions',
	paths  = (pathALCARECOTkAlCosmicsInCollisions),
	content = OutALCARECOTkAlCosmicsInCollisions.outputCommands,
	selectEvents = OutALCARECOTkAlCosmicsInCollisions.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmics = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlCosmics',
	paths  = (pathALCARECOTkAlCosmicsCTF,pathALCARECOTkAlCosmicsCosmicTF,pathALCARECOTkAlCosmicsRegional),
	content = OutALCARECOTkAlCosmics.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmicsHLT = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlCosmicsHLT',
	paths  = (pathALCARECOTkAlCosmicsCTFHLT,pathALCARECOTkAlCosmicsCosmicTFHLT,pathALCARECOTkAlCosmicsRegionalHLT),
	content = OutALCARECOTkAlCosmicsHLT.outputCommands,
	selectEvents = OutALCARECOTkAlCosmicsHLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmics0T = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlCosmics0T',
	paths  = (pathALCARECOTkAlCosmicsCTF0T,pathALCARECOTkAlCosmicsCosmicTF0T,pathALCARECOTkAlCosmicsRegional0T),
	content = OutALCARECOTkAlCosmics0T.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics0T.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmics0THLT = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlCosmics0THLT',
	paths  = (pathALCARECOTkAlCosmicsCTF0THLT,pathALCARECOTkAlCosmicsCosmicTF0THLT,pathALCARECOTkAlCosmicsRegional0THLT),
	content = OutALCARECOTkAlCosmics0THLT.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics0THLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlGlobalCosmics = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'MuAlGlobalCosmics',
	paths  = (pathALCARECOMuAlGlobalCosmics),
	content = OutALCARECOMuAlGlobalCosmics.outputCommands,
	selectEvents = OutALCARECOMuAlGlobalCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlGlobalCosmicsInCollisions = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'MuAlGlobalCosmicsInCollisions',
	paths  = (pathALCARECOMuAlGlobalCosmicsInCollisions),
	content = OutALCARECOMuAlGlobalCosmicsInCollisions.outputCommands,
	selectEvents = OutALCARECOMuAlGlobalCosmicsInCollisions.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlBeamHalo = cms.FilteredStream(
	responsible = 'Andreas Mussgiller',
	name = 'TkAlBeamHalo',
	paths  = (pathALCARECOTkAlBeamHalo),
	content = OutALCARECOTkAlBeamHalo.outputCommands,
	selectEvents = OutALCARECOTkAlBeamHalo.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlBeamHalo = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'MuAlBeamHalo',
	paths  = (pathALCARECOMuAlBeamHalo),
	content = OutALCARECOMuAlBeamHalo.outputCommands,
	selectEvents = OutALCARECOMuAlBeamHalo.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlBeamHaloOverlaps = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'MuAlBeamHaloOverlaps',
	paths  = (pathALCARECOMuAlBeamHaloOverlaps),
	content = OutALCARECOMuAlBeamHaloOverlaps.outputCommands,
	selectEvents = OutALCARECOMuAlBeamHaloOverlaps.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlLAS = cms.FilteredStream(
	responsible = 'Jan Olzem',
	name = 'TkAlLAS',
	paths  = (pathALCARECOTkAlLAS),
	content = OutALCARECOTkAlLAS.outputCommands,
	selectEvents = OutALCARECOTkAlLAS.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)


ALCARECOStreamPromptCalibProd = cms.FilteredStream(
	responsible = 'Gianluca Cerminara',
	name = 'PromptCalibProd',
	paths  = (pathALCARECOPromptCalibProd),
	content = OutALCARECOPromptCalibProd.outputCommands,
	selectEvents = OutALCARECOPromptCalibProd.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiStripPCLHistos = cms.FilteredStream(
	responsible = 'Gianluca Cerminara',
	name = 'SiStripPCLHistos',
	paths  = (pathALCARECOSiStripPCLHistos),
	content = OutALCARECOSiStripPCLHistos.outputCommands,
	selectEvents = OutALCARECOSiStripPCLHistos.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)


from Configuration.StandardSequences.AlCaRecoStream_Specials_cff import *
