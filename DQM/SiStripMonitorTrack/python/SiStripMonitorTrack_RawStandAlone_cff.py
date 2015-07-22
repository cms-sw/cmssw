import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
# Raw Digis
SiStripMonitorTrack.RawDigis_On     = True
SiStripMonitorTrack.RawDigiProducer = 'simSiStripDigis'
SiStripMonitorTrack.RawDigiLabel    = 'VirginRaw'
#
SiStripMonitorTrack.TrackProducer = 'TrackRefitter'
SiStripMonitorTrack.TrackLabel    = ''
SiStripMonitorTrack.Cluster_src = 'siStripClusters'
SiStripMonitorTrack.Mod_On        = False
SiStripMonitorTrack.OffHisto_On   = False
SiStripMonitorTrack.Trend_On      = False
#SiStripMonitorTrack.CCAnalysis_On = True

#### TrackInfo ####
from RecoTracker.TrackProducer.TrackRefitters_cff import *

#-----------------------
#  Reconstruction Modules
#-----------------------
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_SimData_cfi import *


# Digitizer in Virgin Raw Mode
#   Particle properties service
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
# Random numbers initialization service
RandomNumberGeneratorService = cms.Service(
    "RandomNumberGeneratorService",
    VtxSmeared = cms.PSet( initialSeed = cms.untracked.uint32(98765432),
                           engineName = cms.untracked.string('HepJamesRandom')
                           ),
    g4SimHits = cms.PSet( initialSeed = cms.untracked.uint32(11),
                          engineName = cms.untracked.string('HepJamesRandom')
                          ),
    mix = cms.PSet( initialSeed = cms.untracked.uint32(12345),
                    engineName = cms.untracked.string('HepJamesRandom')
                    ),
    simMuonCSCDigis = cms.PSet( initialSeed = cms.untracked.uint32(11223344),
                                engineName = cms.untracked.string('HepJamesRandom')
                                ),
    simMuonDTDigis = cms.PSet( initialSeed = cms.untracked.uint32(1234567),
                               engineName = cms.untracked.string('HepJamesRandom')
                               ),
    simMuonRPCDigis = cms.PSet( initialSeed = cms.untracked.uint32(1234567),
                                engineName = cms.untracked.string('HepJamesRandom')
                                )
    )
# Mixing Module No PileUp
from SimGeneral.MixingModule.mixNoPU_cfi import *
# SiStrip Digitizer
from SimTracker.SiStripDigitizer.SiStripDigi_APVModePeak_cff import *
#simSiStripDigis.ZeroSuppression     = False
#simSiStripDigis.NoiseSigmaThreshold = 0
mix.digitizers.strip.ZeroSuppression = False
mix.digitizers.strip.NoiseSigmaThreshold = 0
#

DQMSiStripMonitorTrack_RawSim = cms.Sequence( mix
                                              *
                                              siStripZeroSuppression
                                              *
                                              TrackRefitter
                                              *
                                              SiStripMonitorTrack
                                              )

# reconstruction sequence for Cosmics
from Configuration.StandardSequences.ReconstructionCosmics_cff import *

DQMSiStripMonitorTrack_CosmicRawSim = cms.Sequence( mix
                                                    *
                                                    trackerCosmics
                                                    *
                                                    TrackRefitter
                                                    *
                                                    SiStripMonitorTrack
                                                    )
