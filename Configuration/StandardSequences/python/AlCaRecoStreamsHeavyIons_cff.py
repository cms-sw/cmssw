import FWCore.ParameterSet.Config as cms

# last update: $Date: 2011/10/11 12:42:35 $ by $Author: cerminar $

# AlCaReco sequence definitions:

###############################################################
# Tracker Alignment
###############################################################
# AlCaReco for track based alignment using ZMuMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlZMuMuHI_cff import *
# AlCaReco for track based alignment using ZMuMu and primary vertex events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlDiMuonAndVertex_cff import *
# AlCaReco for track based alignment using Cosmic muon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsInCollisions_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsHLT_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0THLT_cff import *
# AlCaReco for track based alignment using isoMu events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMuonIsolatedHI_cff import *
# AlCaReco for track based alignment using J/Psi events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlJpsiMuMuHI_cff import *
# AlCaReco for track based alignment using Upsilon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlUpsilonMuMuHI_cff import *
# AlCaReco for track based alignment using MinBias events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlMinBiasHI_cff import *

###############################################################
# Tracker Calibration
###############################################################
# AlCaReco for pixel calibration using muons
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelCalSingleMuon_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelCalSingleMuonLoose_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiPixelCalSingleMuonTight_cff import *
# AlCaReco for tracker calibration using MinBias events
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBiasHI_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalMinBiasAAGHI_cff import *
# AlCaReco for tracker calibration using ZeroBias events (noise)
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalZeroBiasHI_cff import *
# AlCaReco for SiPixel Bad Components using ZeroBias events
from CalibTracker.SiPixelQuality.ALCARECOSiPixelCalZeroBias_cff import *
# AlCaReco for tracker calibration using Cosmics events 
from Calibration.TkAlCaRecoProducers.ALCARECOSiStripCalCosmics_cff import *

###############################################################
# ECAL Calibration
###############################################################

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
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkFilter_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsoTrkProducerFilter_cff import *
# HCAL noise
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalNoise_cff import *
# HCAL isolated bunch
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalIsolatedBunchSelector_cff import *
# HCAL calibration with muons in HB/HE
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHBHEMuonFilter_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHBHEMuonProducerFilter_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalLowPUHBHEMuonFilter_cff import *
from Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalHEMuonFilter_cff import *

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
# nonbeam alcas
###############################################################
from Alignment.CommonAlignmentProducer.ALCARECOTkAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlLAS_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHalo_cff import *
from Alignment.CommonAlignmentProducer.ALCARECOMuAlBeamHaloOverlaps_cff import *

###############################################################
# stream for prompt-calibration @ Tier0
###############################################################
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdHI_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStrip_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGains_cff import *
from Calibration.TkAlCaRecoProducers.ALCARECOPromptCalibProdSiStripGainsAAG_cff import *

from Calibration.TkAlCaRecoProducers.ALCARECOSiStripPCLHistos_cff import *
# FIXME: this needs to be adapted to run on TkAlMinBiasHI tracks
from Alignment.CommonAlignmentProducer.ALCARECOPromptCalibProdSiPixelAli_cff import *

from CalibTracker.SiPixelQuality.ALCARECOPromptCalibProdSiPixel_cff import *



# NOTE: the ALCARECO DQM modules can not be placed together in a single path 
# because the non-DQM sequences act as filters.
# They are therefore inserted per ALCARECO path.
from DQMOffline.Configuration.AlCaRecoDQMHI_cff import *

# AlCaReco path definitions:

pathALCARECOTkAlZMuMuHI = cms.Path(seqALCARECOTkAlZMuMuHI*ALCARECOTkAlZMuMuHIDQM)
pathALCARECOTkAlDiMuonAndVertex = cms.Path(seqALCARECOTkAlDiMuonAndVertex)
pathALCARECOTkAlMuonIsolatedHI = cms.Path(seqALCARECOTkAlMuonIsolatedHI*ALCARECOTkAlMuonIsolatedHIDQM)
pathALCARECOTkAlJpsiMuMuHI = cms.Path(seqALCARECOTkAlJpsiMuMuHI*ALCARECOTkAlJpsiMuMuHIDQM)
pathALCARECOTkAlUpsilonMuMuHI = cms.Path(seqALCARECOTkAlUpsilonMuMuHI*ALCARECOTkAlUpsilonMuMuHIDQM)
pathALCARECOTkAlMinBiasHI = cms.Path(seqALCARECOTkAlMinBiasHI*ALCARECOTkAlMinBiasHIDQM)
pathALCARECOSiPixelCalSingleMuon = cms.Path(seqALCARECOSiPixelCalSingleMuon)
pathALCARECOSiPixelCalSingleMuonLoose = cms.Path(seqALCARECOSiPixelCalSingleMuonLoose)
pathALCARECOSiPixelCalSingleMuonTight = cms.Path(seqALCARECOSiPixelCalSingleMuonTight)
pathALCARECOSiStripCalMinBias = cms.Path(seqALCARECOSiStripCalMinBias*ALCARECOSiStripCalMinBiasDQM)
pathALCARECOSiStripCalMinBiasAAG = cms.Path(seqALCARECOSiStripCalMinBiasAAG*ALCARECOSiStripCalMinBiasAAGDQM)
pathALCARECOSiStripCalCosmics = cms.Path(seqALCARECOSiStripCalCosmics)
pathALCARECOSiStripCalZeroBias = cms.Path(seqALCARECOSiStripCalZeroBias*ALCARECOSiStripCalZeroBiasDQM)
pathALCARECOSiPixelCalZeroBias = cms.Path(seqALCARECOSiPixelCalZeroBias)

pathALCARECOHcalCalDijets = cms.Path(seqALCARECOHcalCalDijets*ALCARECOHcalCalDiJetsDQM)
pathALCARECOHcalCalGammaJet = cms.Path(seqALCARECOHcalCalGammaJet)
pathALCARECOHcalCalHO = cms.Path(seqALCARECOHcalCalHO*ALCARECOHcalCalHODQM)
pathALCARECOHcalCalHOCosmics = cms.Path(seqALCARECOHcalCalHOCosmics)
pathALCARECOHcalCalIsoTrk = cms.Path(seqALCARECOHcalCalIsoTrk*ALCARECOHcalCalIsoTrackDQM)
pathALCARECOHcalCalIsoTrkFilter = cms.Path(seqALCARECOHcalCalIsoTrkFilter)
pathALCARECOHcalCalIsoTrkProducerFilter = cms.Path(seqALCARECOHcalCalIsoTrkProducerFilter)
pathALCARECOHcalCalNoise = cms.Path(seqALCARECOHcalCalNoise)
pathALCARECOHcalCalIsolatedBunchSelector = cms.Path(seqALCARECOHcalCalIsolatedBunchSelector*ALCARECOHcalCalIsolatedBunchDQM)
pathALCARECOHcalCalHBHEMuonFilter = cms.Path(seqALCARECOHcalCalHBHEMuonFilter)
pathALCARECOHcalCalHBHEMuonProducerFilter = cms.Path(seqALCARECOHcalCalHBHEMuonProducerFilter)
pathALCARECOHcalCalLowPUHBHEMuonFilter = cms.Path(seqALCARECOHcalCalLowPUHBHEMuonFilter)
pathALCARECOHcalCalHEMuonFilter = cms.Path(seqALCARECOHcalCalHEMuonFilter)
pathALCARECOMuAlCalIsolatedMu = cms.Path(seqALCARECOMuAlCalIsolatedMu)
pathALCARECOMuAlZMuMu = cms.Path(seqALCARECOMuAlZMuMu)
pathALCARECOMuAlOverlaps = cms.Path(seqALCARECOMuAlOverlaps)
pathALCARECORpcCalHLT = cms.Path(seqALCARECORpcCalHLT)
pathALCARECOTkAlBeamHalo = cms.Path(seqALCARECOTkAlBeamHalo*ALCARECOTkAlBeamHaloDQM)
pathALCARECOMuAlBeamHaloOverlaps = cms.Path(seqALCARECOMuAlBeamHaloOverlaps)
pathALCARECOMuAlBeamHalo = cms.Path(seqALCARECOMuAlBeamHalo)
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
pathALCARECOMuAlGlobalCosmicsInCollisions = cms.Path(seqALCARECOMuAlGlobalCosmicsInCollisions)
pathALCARECOMuAlGlobalCosmics = cms.Path(seqALCARECOMuAlGlobalCosmics)
pathALCARECOPromptCalibProd = cms.Path(seqALCARECOPromptCalibProd)
pathALCARECOPromptCalibProdSiStrip = cms.Path(seqALCARECOPromptCalibProdSiStrip)
pathALCARECOPromptCalibProdSiStripGains = cms.Path(seqALCARECOPromptCalibProdSiStripGains)
pathALCARECOPromptCalibProdSiStripGainsAAG = cms.Path(seqALCARECOPromptCalibProdSiStripGainsAAG)
pathALCARECOPromptCalibProdSiPixelAli = cms.Path(seqALCARECOPromptCalibProdSiPixelAli)
pathALCARECOPromptCalibProdSiPixel = cms.Path(seqALCARECOPromptCalibProdSiPixel)
pathALCARECOSiStripPCLHistos = cms.Path(seqALCARECOSiStripPCLHistos)

# AlCaReco event content definitions:

from Configuration.EventContent.AlCaRecoOutput_cff import *

# AlCaReco stream definitions:


ALCARECOStreamTkAlMinBiasHI = cms.FilteredStream(
	responsible = 'James Castle',
	name = 'TkAlMinBiasHI',
	paths  = (pathALCARECOTkAlMinBiasHI),
	content = OutALCARECOTkAlMinBiasHI.outputCommands,
	selectEvents = OutALCARECOTkAlMinBiasHI.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlMuonIsolatedHI = cms.FilteredStream(
	responsible = 'James Castle',
	name = 'TkAlMuonIsolatedHI',
	paths  = (pathALCARECOTkAlMuonIsolatedHI),
	content = OutALCARECOTkAlMuonIsolatedHI.outputCommands,
	selectEvents = OutALCARECOTkAlMuonIsolatedHI.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlZMuMuHI = cms.FilteredStream(
	responsible = 'James Castle',
	name = 'TkAlZMuMuHI',
	paths  = (pathALCARECOTkAlZMuMuHI),
	content = OutALCARECOTkAlZMuMuHI.outputCommands,
	selectEvents = OutALCARECOTkAlZMuMuHI.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlDiMuonAndVertex = cms.FilteredStream(
	responsible = 'Marco Musich',
	name = 'TkAlDiMuonAndVertex',
	paths  = (pathALCARECOTkAlDiMuonAndVertex),
	content = OutALCARECOTkAlDiMuonAndVertex.outputCommands,
	selectEvents = OutALCARECOTkAlDiMuonAndVertex.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlJpsiMuMuHI = cms.FilteredStream(
	responsible = 'James Castle',
	name = 'TkAlJpsiMuMuHI',
	paths  = (pathALCARECOTkAlJpsiMuMuHI),
	content = OutALCARECOTkAlJpsiMuMuHI.outputCommands,
	selectEvents = OutALCARECOTkAlJpsiMuMuHI.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamTkAlUpsilonMuMuHI = cms.FilteredStream(
	responsible = 'James Castle',
	name = 'TkAlUpsilonMuMuHI',
	paths  = (pathALCARECOTkAlUpsilonMuMuHI),
	content = OutALCARECOTkAlUpsilonMuMuHI.outputCommands,
	selectEvents = OutALCARECOTkAlUpsilonMuMuHI.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiPixelCalSingleMuon = cms.FilteredStream(
	responsible = 'Tamas Almos Vami',
	name = 'SiPixelCalSingleMuon',
	paths  = (pathALCARECOSiPixelCalSingleMuon),
	content = OutALCARECOSiPixelCalSingleMuon.outputCommands,
	selectEvents = OutALCARECOSiPixelCalSingleMuon.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiPixelCalSingleMuonTight = cms.FilteredStream(
	responsible = 'Marco Musich',
	name = 'SiPixelCalSingleMuonTight',
	paths  = (pathALCARECOSiPixelCalSingleMuonTight),
	content = OutALCARECOSiPixelCalSingleMuonTight.outputCommands,
	selectEvents = OutALCARECOSiPixelCalSingleMuonTight.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamSiPixelCalSingleMuonLoose = cms.FilteredStream(
	responsible = 'Marco Musich',
	name = 'SiPixelCalSingleMuonLoose',
	paths  = (pathALCARECOSiPixelCalSingleMuonLoose),
	content = OutALCARECOSiPixelCalSingleMuonLoose.outputCommands,
	selectEvents = OutALCARECOSiPixelCalSingleMuonLoose.SelectEvents,
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

ALCARECOStreamSiStripCalMinBiasAAG = cms.FilteredStream(
        responsible = 'Alessandro Di Mattia',
        name = 'SiStripCalMinBiasAAG',
        paths  = (pathALCARECOSiStripCalMinBiasAAG),
        content = OutALCARECOSiStripCalMinBiasAAG.outputCommands,
        selectEvents = OutALCARECOSiStripCalMinBiasAAG.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

ALCARECOStreamSiStripCalCosmics = cms.FilteredStream(
	responsible = 'Marco Musich',
	name = 'SiStripCalCosmics',
	paths  = (pathALCARECOSiStripCalCosmics),
	content = OutALCARECOSiStripCalCosmics.outputCommands,
	selectEvents = OutALCARECOSiStripCalCosmics.SelectEvents,
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

ALCARECOStreamSiPixelCalZeroBias = cms.FilteredStream(
        responsible = 'Tongguang Cheng',
        name = 'SiPixelCalZeroBias',
        paths  = (pathALCARECOSiPixelCalZeroBias),
        content = OutALCARECOSiPixelCalZeroBias.outputCommands,
        selectEvents = OutALCARECOSiPixelCalZeroBias.SelectEvents,
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
	responsible = 'Sunanda Banerjee',
	name = 'HcalCalIsoTrk',
	paths  = (pathALCARECOHcalCalIsoTrk),
	content = OutALCARECOHcalCalIsoTrk.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrk.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalIsoTrkFilter = cms.FilteredStream(
	responsible = 'Sunanda Banerjee',
	name = 'HcalCalIsoTrkFilter',
	paths  = (pathALCARECOHcalCalIsoTrkFilter),
	content = OutALCARECOHcalCalIsoTrkFilter.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrkFilter.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalIsoTrkProducerFilter = cms.FilteredStream(
	responsible = 'Sunanda Banerjee',
	name = 'HcalCalIsoTrkProducerFilter',
	paths  = (pathALCARECOHcalCalIsoTrkProducerFilter),
	content = OutALCARECOHcalCalIsoTrkProducerFilter.outputCommands,
	selectEvents = OutALCARECOHcalCalIsoTrkProducerFilter.SelectEvents,
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

ALCARECOStreamHcalCalIsolatedBunchSelector = cms.FilteredStream(
	responsible = 'Sunanda Banerjee',
	name = 'HcalCalIsolatedBunchSelector',
	paths  = (pathALCARECOHcalCalIsolatedBunchSelector),
	content = OutALCARECOHcalCalIsolatedBunchSelector.outputCommands,
	selectEvents = OutALCARECOHcalCalIsolatedBunchSelector.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalHBHEMuonFilter = cms.FilteredStream(
	responsible = 'Sunanda Banerjee',
	name = 'HcalCalHBHEMuonFilter',
	paths  = (pathALCARECOHcalCalHBHEMuonFilter),
	content = OutALCARECOHcalCalHBHEMuonFilter.outputCommands,
	selectEvents = OutALCARECOHcalCalHBHEMuonFilter.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalHBHEMuonProducerFilter = cms.FilteredStream(
	responsible = 'Sunanda Banerjee',
	name = 'HcalCalHBHEMuonProducerFilter',
	paths  = (pathALCARECOHcalCalHBHEMuonProducerFilter),
	content = OutALCARECOHcalCalHBHEMuonProducerFilter.outputCommands,
	selectEvents = OutALCARECOHcalCalHBHEMuonProducerFilter.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalLowPUHBHEMuonFilter = cms.FilteredStream(
	responsible = 'Nan Lu',
	name = 'HcalCalLowPUHBHEMuonFilter',
	paths  = (pathALCARECOHcalCalLowPUHBHEMuonFilter),
	content = OutALCARECOHcalCalLowPUHBHEMuonFilter.outputCommands,
	selectEvents = OutALCARECOHcalCalLowPUHBHEMuonFilter.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamHcalCalHEMuonFilter = cms.FilteredStream(
	responsible = 'Nan Lu',
	name = 'HcalCalHEMuonFilter',
	paths  = (pathALCARECOHcalCalHEMuonFilter),
	content = OutALCARECOHcalCalHEMuonFilter.outputCommands,
	selectEvents = OutALCARECOHcalCalHEMuonFilter.SelectEvents,
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

ALCARECOStreamPromptCalibProdSiStrip = cms.FilteredStream(
	responsible = 'Gianluca Cerminara',
	name = 'PromptCalibProdSiStrip',
	paths  = (pathALCARECOPromptCalibProdSiStrip),
	content = OutALCARECOPromptCalibProdSiStrip.outputCommands,
	selectEvents = OutALCARECOPromptCalibProdSiStrip.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)


ALCARECOStreamPromptCalibProdSiStripGains = cms.FilteredStream(
	responsible = 'Gianluca Cerminara',
	name = 'PromptCalibProdSiStripGains',
	paths  = (pathALCARECOPromptCalibProdSiStripGains),
	content = OutALCARECOPromptCalibProdSiStripGains.outputCommands,
	selectEvents = OutALCARECOPromptCalibProdSiStripGains.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamPromptCalibProdSiStripGainsAAG = cms.FilteredStream(
        responsible = 'Alessandro Di Mattia',
        name = 'PromptCalibProdSiStripGainsAAG',
        paths  = (pathALCARECOPromptCalibProdSiStripGainsAAG),
        content = OutALCARECOPromptCalibProdSiStripGainsAAG.outputCommands,
        selectEvents = OutALCARECOPromptCalibProdSiStripGainsAAG.SelectEvents,
        dataTier = cms.untracked.string('ALCARECO')
        )

ALCARECOStreamPromptCalibProdSiPixelAli = cms.FilteredStream(
	responsible = 'Gianluca Cerminara',
	name = 'PromptCalibProdSiPixelAli',
	paths  = (pathALCARECOPromptCalibProdSiPixelAli),
	content = OutALCARECOPromptCalibProdSiPixelAli.outputCommands,
	selectEvents = OutALCARECOPromptCalibProdSiPixelAli.SelectEvents,
	dataTier = cms.untracked.string('ALCARECO')
	)

ALCARECOStreamPromptCalibProdSiPixel = cms.FilteredStream(
        responsible = 'Tongguang Cheng',
        name = 'PromptCalibProdSiPixel',
        paths  = (pathALCARECOPromptCalibProdSiPixel),
        content = OutALCARECOPromptCalibProdSiPixel.outputCommands,
        selectEvents = OutALCARECOPromptCalibProdSiPixel.SelectEvents,
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


from Configuration.StandardSequences.AlCaRecoStream_SpecialsHI_cff import *

