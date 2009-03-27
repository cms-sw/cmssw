import FWCore.ParameterSet.Config as cms

import copy
# from RecoTracker.TrackProducer.RefitterWithMaterial_cfi import *

process = cms.Process("RECO3")

process.load("FWCore/MessageService/MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Simulation_cff")

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("TrackingTools.TrackAssociator.default_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

# track refitting to have local hit position which is not stored on disk (from Giuseppe Cerati)
# process.RefitTracks = copy.deepcopy(TrackRefitter)
# process.RefitTracks.src = cms.InputTag("generalTracks")

process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEGeneric_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoPixelVertexing.PixelTrackFitting.PixelTracks_cff")

process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.TrackRefitter.TrajectoryInEvent = True

#add highPurity quality bit
process.load("RecoTracker.FinalTrackSelectors.selectHighPurity_cfi")
process.selectHighPurity.src = cms.InputTag('TrackRefitter')
process.selectHighPurity.copyExtras = True
process.selectHighPurity.copyTrajectories = True
#set this to false to have all tracks (highQuality or not but with quality added). true makes the
#collection contain only highQuality tracks
process.selectHighPurity.keepAllTracks = True

# switch OFF ECAL SR
from SimCalorimetry.EcalSimProducers.ecaldigi_cfi import *
from SimCalorimetry.EcalSelectiveReadoutProducers.ecalDigis_cfi import *
simEcalDigis.srpBarrelLowInterestChannelZS = cms.double(-1.e9)
simEcalDigis.srpEndcapLowInterestChannelZS = cms.double(-1.e9)

from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
ecalWeightUncalibRecHit.EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis")
ecalWeightUncalibRecHit.EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis")
#

#
process.load("SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi")
#

# HCAL ZSP
from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import *
from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigisRealistic_cfi import *
simHcalDigis.HBlevel = cms.int32(-8000)
simHcalDigis.HElevel = cms.int32(-8000)
simHcalDigis.HOlevel = cms.int32(-8000)
simHcalDigis.HFlevel = cms.int32(-9000)

#from SimCalorimetry.HcalZeroSuppressionProducers.hcalDigis_cfi import *
#simHcalDigis.hbhe = cms.PSet(
#        firstSample = cms.int32(4),
#        mode = cms.int32(2),
#        samplesToAdd = cms.int32(2),
#        twoSided = cms.bool(False),
#        level = cms.int32(-2000)
#    )

from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi import *
hbhereco.digiLabel = cms.InputTag("simHcalDigis")

from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi import *
hfreco.digiLabel = cms.InputTag("simHcalDigis")

from RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi import *
horeco.digiLabel = cms.InputTag("simHcalDigis")
#

#
# test QCD file from 210 RelVal is on /castor/cern.ch/user/a/anikiten/jpt210qcdfile/
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source("PoolSource",
# cmssw210
#    fileNames = cms.untracked.vstring('file:/tmp/anikiten/FC999068-DB60-DD11-9694-001A92971B16.root')
# cmssw218
    fileNames = cms.untracked.vstring('file:/tmp/anikiten/PARTICLEGUN_DIPION_50_FULLCALO_cff_py_RAW2DIGI_RECO.root')
)

process.dump = cms.EDFilter("EventContentAnalyzer")

#    tracks = cms.string('RefitTracks'), 
process.myanalysis = cms.EDFilter("SinglePionEfficiencyNew",
    HistOutFile = cms.untracked.string('SinglePionEfficiencyNew.root'),
    tracks = cms.string('selectHighPurity'), 
    pxltracks = cms.string('pixelTracks'),
    pxlhits = cms.string('siPixelRecHits'),
    calotowers = cms.string('caloTowers'),
    towermaker = cms.string('towerMaker'),
    hbheInput = cms.string('hbhereco'),
    hoInput = cms.string('horeco'),
    hfInput = cms.string('hfreco'),
    ecalRecHitsProducer = cms.string('ecalRecHit'),
    ECALbarrelHitCollection = cms.string('EcalRecHitsEB'),
    ECALendcapHitCollection = cms.string('EcalRecHitsEE'),
    TrackAssociatorParameters = cms.PSet(
      muonMaxDistanceSigmaX = cms.double(0.0),
      muonMaxDistanceSigmaY = cms.double(0.0),
      CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
      dRHcal = cms.double(9999.0),
      dREcal = cms.double(9999.0),
      CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
      useEcal = cms.bool(True),
      dREcalPreselection = cms.double(0.05),
      HORecHitCollectionLabel = cms.InputTag("horeco"),
      dRMuon = cms.double(9999.0),
      crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
      muonMaxDistanceX = cms.double(5.0),
      muonMaxDistanceY = cms.double(5.0),
      useHO = cms.bool(False),
      accountForTrajectoryChangeCalo = cms.bool(False),
      DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
      EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
      dRHcalPreselection = cms.double(0.2),
      useMuon = cms.bool(False),
      useCalo = cms.bool(True),
      EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
      dRMuonPreselection = cms.double(0.2),
      truthMatch = cms.bool(False),
      HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
      useHcal = cms.bool(False))
)


process.dump = cms.EDFilter("EventContentAnalyzer")

# process.p1 = cms.Path(process.mix*process.dump)
# process.p1 = cms.Path(process.mix*process.simHcalUnsuppressedDigis*process.simHcalDigis*process.hbhereco*process.hfreco*process.horeco*process.dump)

# ECAL SR, HCAL ZS  OFF

#process.p1 = cms.Path(process.TrackRefitter*process.siPixelRecHits*process.pixelTracks*process.selectHighPurity*process.dump)

process.p1 = cms.Path(process.mix*process.TrackRefitter*process.siPixelRecHits*process.pixelTracks*process.selectHighPurity*process.simEcalUnsuppressedDigis*process.simEcalDigis*process.ecalWeightUncalibRecHit*process.ecalRecHit*process.simHcalUnsuppressedDigis*process.simHcalDigis*process.hbhereco*process.myanalysis)


