import FWCore.ParameterSet.Config as cms

# last update: $Date: 2009/03/19 17:04:09 $ by $Author: argiro $

# AlCaReco sequence definitions:

###############################################################
# Tracker Alignment
###############################################################
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
# HCAL calibration with gamma+jet
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_cff import *
# HCAL calibration from HO (muons) 
#  include "Calibration/HcalAlCaRecoProducers/data/ALCARECOHcalCalZMuMu.cff"
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHO_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHOCosmics_cff import *
###############################################################
# Muon alignment
###############################################################
# Muon Alignment with cosmics
from Alignment.CommonAlignmentProducer.ALCARECOMuAlStandAloneCosmics_cff import *
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
# nonbeam alcas
###############################################################
from Alignment.CommonAlignmentProducer.ALCARECOTkAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlLAS_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHaloOverlaps_cff import *

# NOTE: the ALCARECO DQM modules can not be placed together in a single path 
# because the non-DQM sequences act as filters.
# They are therefore inserted per ALCARECO path.
from DQMOffline.Configuration.AlCaRecoDQM_cff import *

# AlCaReco path definitions:

pathALCARECOTkAlZMuMu = cms.Path(seqALCARECOTkAlZMuMu*ALCARECOTkAlZMuMuDQM)
pathALCARECOTkAlMuonIsolated = cms.Path(seqALCARECOTkAlMuonIsolated*ALCARECOTkAlMuonIsolatedDQM)
pathALCARECOTkAlJpsiMuMu = cms.Path(seqALCARECOTkAlJpsiMuMu*ALCARECOTkAlJpsiMuMuDQM)
pathALCARECOTkAlUpsilonMuMu = cms.Path(seqALCARECOTkAlUpsilonMuMu*ALCARECOTkAlUpsilonMuMuDQM)
pathALCARECOTkAlMinBias = cms.Path(seqALCARECOTkAlMinBias*ALCARECOTkAlMinBiasDQM)
pathALCARECOTkAlMinBias = cms.Path(seqALCARECOTkAlMinBias*ALCARECOTkAlMinBiasDQM)
pathALCARECOSiPixelLorentzAngle = cms.Path(seqALCARECOSiPixelLorentzAngle)
pathALCARECOSiStripCalMinBias = cms.Path(seqALCARECOSiStripCalMinBias)
pathALCARECOSiStripCalZeroBias = cms.Path(seqALCARECOSiStripCalZeroBias*ALCARECOSiStripCalZeroBiasDQM)
pathALCARECOEcalCalElectron = cms.Path(seqALCARECOEcalCalElectron*ALCARECOEcalCalElectronCalibDQM)
pathALCARECOHcalCalDijets = cms.Path(seqALCARECOHcalCalDijets)
pathALCARECOHcalCalGammaJet = cms.Path(seqALCARECOHcalCalGammaJet)
pathALCARECOHcalCalHO = cms.Path(seqALCARECOHcalCalHO)
pathALCARECOHcalCalHOCosmics = cms.Path(seqALCARECOHcalCalHOCosmics)
pathALCARECOMuAlCalIsolatedMu = cms.Path(seqALCARECOMuAlCalIsolatedMu*ALCARECOMuAlCalIsolatedMuDQM)
pathALCARECOMuAlZMuMu = cms.Path(seqALCARECOMuAlZMuMu*ALCARECOMuAlZMuMuDQM)
pathALCARECOMuAlOverlaps = cms.Path(seqALCARECOMuAlOverlaps*ALCARECOMuAlOverlapsDQM)
pathALCARECORpcCalHLT = cms.Path(seqALCARECORpcCalHLT)
pathALCARECOTkAlBeamHalo = cms.Path(seqALCARECOTkAlBeamHalo)
pathALCARECOMuAlBeamHaloOverlaps = cms.Path(seqALCARECOMuAlBeamHaloOverlaps*ALCARECOMuAlBeamHaloOverlapsDQM)
pathALCARECOMuAlBeamHalo = cms.Path(seqALCARECOMuAlBeamHalo*ALCARECOMuAlBeamHaloDQM)
pathALCARECOTkAlLAS = cms.Path(seqALCARECOTkAlLAS)
pathALCARECOTkAlCosmicsCTF = cms.Path(seqALCARECOTkAlCosmicsCTF*ALCARECOTkAlCosmicsCTFDQM)
pathALCARECOTkAlCosmicsCosmicTF = cms.Path(seqALCARECOTkAlCosmicsCosmicTF*ALCARECOTkAlCosmicsCosmicTFDQM)
pathALCARECOTkAlCosmicsRS = cms.Path(seqALCARECOTkAlCosmicsRS*ALCARECOTkAlCosmicsRSDQM)
pathALCARECOTkAlCosmicsCTF0T = cms.Path(seqALCARECOTkAlCosmicsCTF0T*ALCARECOTkAlCosmicsCTF0TDQM)
pathALCARECOTkAlCosmicsCosmicTF0T = cms.Path(seqALCARECOTkAlCosmicsCosmicTF0T*ALCARECOTkAlCosmicsCosmicTF0TDQM)
pathALCARECOTkAlCosmicsRS0T = cms.Path(seqALCARECOTkAlCosmicsRS0T*ALCARECOTkAlCosmicsRS0TDQM)
pathALCARECOTkAlCosmicsCTFHLT = cms.Path(seqALCARECOTkAlCosmicsCTFHLT*ALCARECOTkAlCosmicsCTFDQM)
pathALCARECOTkAlCosmicsCosmicTFHLT = cms.Path(seqALCARECOTkAlCosmicsCosmicTFHLT*ALCARECOTkAlCosmicsCosmicTFDQM)
pathALCARECOTkAlCosmicsRSHLT = cms.Path(seqALCARECOTkAlCosmicsRSHLT*ALCARECOTkAlCosmicsRSDQM)
pathALCARECOTkAlCosmicsCTF0THLT = cms.Path(seqALCARECOTkAlCosmicsCTF0THLT*ALCARECOTkAlCosmicsCTF0TDQM)
pathALCARECOTkAlCosmicsCosmicTF0THLT = cms.Path(seqALCARECOTkAlCosmicsCosmicTF0THLT*ALCARECOTkAlCosmicsCosmicTF0TDQM)
pathALCARECOTkAlCosmicsRS0THLT = cms.Path(seqALCARECOTkAlCosmicsRS0THLT*ALCARECOTkAlCosmicsRS0TDQM)
pathALCARECOMuAlStandAloneCosmics = cms.Path(seqALCARECOMuAlStandAloneCosmics*ALCARECOMuAlStandAloneCosmicsDQM)
pathALCARECOMuAlGlobalCosmics = cms.Path(seqALCARECOMuAlGlobalCosmics*ALCARECOMuAlGlobalCosmicsDQM)

# AlCaReco event content definitions:

from Configuration.EventContent.AlCaRecoOutput_cff import *

# AlCaReco stream definitions:

ALCARECOStreamTkAlMinBias = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlMinBias',
	paths  = (pathALCARECOTkAlMinBias),
	content = OutALCARECOTkAlMinBias.outputCommands,
	selectEvents = OutALCARECOTkAlMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlMuonIsolated = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlMuonIsolated',
	paths  = (pathALCARECOTkAlMuonIsolated),
	content = OutALCARECOTkAlMuonIsolated.outputCommands,
	selectEvents = OutALCARECOTkAlMuonIsolated.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlZMuMu = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlZMuMu',
	paths  = (pathALCARECOTkAlZMuMu),
	content = OutALCARECOTkAlZMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlZMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlJpsiMuMu = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlJpsiMuMu',
	paths  = (pathALCARECOTkAlJpsiMuMu),
	content = OutALCARECOTkAlJpsiMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlJpsiMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlUpsilonMuMu = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlUpsilonMuMu',
	paths  = (pathALCARECOTkAlUpsilonMuMu),
	content = OutALCARECOTkAlUpsilonMuMu.outputCommands,
	selectEvents = OutALCARECOTkAlUpsilonMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiPixelLorentzAngle = cms.FilteredStream(
	responsible = 'Lotte Wilke',
	name = 'ALCARECOSiPixelLorentzAngle',
	paths  = (pathALCARECOSiPixelLorentzAngle),
	content = OutALCARECOSiPixelLorentzAngle.outputCommands,
	selectEvents = OutALCARECOSiPixelLorentzAngle.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiStripCalMinBias = cms.FilteredStream(
	responsible = 'Vitaliano Ciulli',
	name = 'ALCARECOSiStripCalMinBias',
	paths  = (pathALCARECOSiStripCalMinBias),
	content = OutALCARECOSiStripCalMinBias.outputCommands,
	selectEvents = OutALCARECOSiStripCalMinBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiStripCalZeroBias = cms.FilteredStream(
	responsible = 'Gordon Kaussen',
	name = 'ALCARECOSiStripCalZeroBias',
	paths  = (pathALCARECOSiStripCalZeroBias),
	content = OutALCARECOSiStripCalZeroBias.outputCommands,
	selectEvents = OutALCARECOSiStripCalZeroBias.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamEcalCalElectron = cms.FilteredStream(
	responsible = 'Pietro Govoni',
	name = 'ALCARECOEcalCalElectron',
	paths  = (pathALCARECOEcalCalElectron),
	content = OutALCARECOEcalCalElectron.outputCommands,
	selectEvents = OutALCARECOEcalCalElectron.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalDijets = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalDijets',
	paths  = (pathALCARECOHcalCalDijets),
	content = OutALCARECOHcalCalDijets.outputCommands,
	selectEvents = OutALCARECOHcalCalDijets.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalGammaJet = cms.FilteredStream(
	responsible = 'Grigory Safronov',
	name = 'ALCARECOHcalCalGammaJet',
	paths  = (pathALCARECOHcalCalGammaJet),
	content = OutALCARECOHcalCalGammaJet.outputCommands,
	selectEvents = OutALCARECOHcalCalGammaJet.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalHO = cms.FilteredStream(
	responsible = 'Gobinda Majumder',
	name = 'ALCARECOHcalCalHO',
	paths  = (pathALCARECOHcalCalHO),
	content = OutALCARECOHcalCalHO.outputCommands,
	selectEvents = OutALCARECOHcalCalHO.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalHOCosmics = cms.FilteredStream(
	responsible = 'Gobinda Majumder',
	name = 'ALCARECOHcalCalHOCosmics',
	paths  = (pathALCARECOHcalCalHOCosmics),
	content = OutALCARECOHcalCalHOCosmics.outputCommands,
	selectEvents = OutALCARECOHcalCalHOCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlCalIsolatedMu = cms.FilteredStream(
	responsible = 'Javier Fernandez',
	name = 'ALCARECOMuAlCalIsolatedMu',
	paths  = (pathALCARECOMuAlCalIsolatedMu),
	content = OutALCARECOMuAlCalIsolatedMu.outputCommands,
	selectEvents = OutALCARECOMuAlCalIsolatedMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlZMuMu = cms.FilteredStream(
	responsible = 'Javier Fernandez',
	name = 'ALCARECOMuAlZMuMu',
	paths  = (pathALCARECOMuAlZMuMu),
	content = OutALCARECOMuAlZMuMu.outputCommands,
	selectEvents = OutALCARECOMuAlZMuMu.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlOverlaps = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlOverlaps',
	paths  = (pathALCARECOMuAlOverlaps),
	content = OutALCARECOMuAlOverlaps.outputCommands,
	selectEvents = OutALCARECOMuAlOverlaps.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamRpcCalHLT = cms.FilteredStream(
	responsible = 'Marcello Maggi',
	name = 'ALCARECORpcCalHLT',
	paths  = (pathALCARECORpcCalHLT),
	content = OutALCARECORpcCalHLT.outputCommands,
	selectEvents = OutALCARECORpcCalHLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmics = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmics',
	paths  = (pathALCARECOTkAlCosmicsCTF,pathALCARECOTkAlCosmicsCosmicTF,pathALCARECOTkAlCosmicsRS),
	content = OutALCARECOTkAlCosmics.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmicsHLT = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmicsHLT',
	paths  = (pathALCARECOTkAlCosmicsCTFHLT,pathALCARECOTkAlCosmicsCosmicTFHLT,pathALCARECOTkAlCosmicsRSHLT),
	content = OutALCARECOTkAlCosmicsHLT.outputCommands,
	selectEvents = OutALCARECOTkAlCosmicsHLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmics0T = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmics0T',
	paths  = (pathALCARECOTkAlCosmicsCTF0T,pathALCARECOTkAlCosmicsCosmicTF0T,pathALCARECOTkAlCosmicsRS0T),
	content = OutALCARECOTkAlCosmics0T.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics0T.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlCosmics0THLT = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlCosmics0THLT',
	paths  = (pathALCARECOTkAlCosmicsCTF0THLT,pathALCARECOTkAlCosmicsCosmicTF0THLT,pathALCARECOTkAlCosmicsRS0THLT),
	content = OutALCARECOTkAlCosmics0THLT.outputCommands,
	selectEvents = OutALCARECOTkAlCosmics0THLT.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlStandAloneCosmics = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlStandAloneCosmics',
	paths  = (pathALCARECOMuAlStandAloneCosmics),
	content = OutALCARECOMuAlStandAloneCosmics.outputCommands,
	selectEvents = OutALCARECOMuAlStandAloneCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlGlobalCosmics = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlGlobalCosmics',
	paths  = (pathALCARECOMuAlGlobalCosmics),
	content = OutALCARECOMuAlGlobalCosmics.outputCommands,
	selectEvents = OutALCARECOMuAlGlobalCosmics.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlBeamHalo = cms.FilteredStream(
	responsible = 'Gero Flucke',
	name = 'ALCARECOTkAlBeamHalo',
	paths  = (pathALCARECOTkAlBeamHalo),
	content = OutALCARECOTkAlBeamHalo.outputCommands,
	selectEvents = OutALCARECOTkAlBeamHalo.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlBeamHalo = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlBeamHalo',
	paths  = (pathALCARECOMuAlBeamHalo),
	content = OutALCARECOMuAlBeamHalo.outputCommands,
	selectEvents = OutALCARECOMuAlBeamHalo.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamMuAlBeamHaloOverlaps = cms.FilteredStream(
	responsible = 'Jim Pivarski',
	name = 'ALCARECOMuAlBeamHaloOverlaps',
	paths  = (pathALCARECOMuAlBeamHaloOverlaps),
	content = OutALCARECOMuAlBeamHaloOverlaps.outputCommands,
	selectEvents = OutALCARECOMuAlBeamHaloOverlaps.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlLAS = cms.FilteredStream(
	responsible = 'Jan Olzem',
	name = 'ALCARECOTkAlLAS',
	paths  = (pathALCARECOTkAlLAS),
	content = OutALCARECOTkAlLAS.outputCommands,
	selectEvents = OutALCARECOTkAlLAS.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)
