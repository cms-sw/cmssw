import FWCore.ParameterSet.Config as cms

process = cms.Process("LA")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("RecoTracker.Configuration.RecoTracker_cff")

process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.TrackRefitter.src = "generalTracks"
# process.TrackRefitter.src = "globalMuons"
process.TrackRefitter.TrajectoryInEvent = True
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
process.offlineBeamSpot = offlineBeamSpot

# process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V6::All'


process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('simul', 
        'cout'),
    simul = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:2E5CADBF-7B6C-DD11-9888-0019DB29C614.root')
# 	  fileNames = cms.untracked.vstring('file:/home/wilke/CMSSW_2_1_4/src/ALCARECOSiPixelLorentzAngle.root')	
)

process.lorentzAngle = cms.EDAnalyzer("SiPixelLorentzAngle",
	TTRHBuilder= cms.string("WithTrackAngle"),
	Fitter = cms.string("KFFittingSmoother"),
  Propagator = cms.string("PropagatorWithMaterial"),
#what type of tracks should be used: 
	src = cms.string("TrackRefitter"),
#   src = cms.string("globalMuons"),
# src = cms.string("ALCARECOTkAlMinBias"),
	fileName = cms.string("lorentzangle.root"),
	fileNameFit	= cms.string("lorentzFit.txt"),
	binsDepth	= cms.int32(50),
	binsDrift =	cms.int32(60),
	ptMin = cms.double(3),
	#in case of MC set this to true to save the simhits
	simData = cms.bool(False)
)

process.p = cms.Path(process.offlineBeamSpot*process.TrackRefitter*process.lorentzAngle)
