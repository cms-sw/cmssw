import FWCore.ParameterSet.Config as cms

process = cms.Process("LA")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.ReconstructionCosmics_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "frontier://FrontierProd/CMS_COND_21X_GLOBALTAG"
process.GlobalTag.globaltag = "CRAFT_V3P::All"
# process.GlobalTag.globaltag = "CRAFT_ALL_V3::All"
# process.es_prefer_GlobalTag = cms.ESPrefer('PoolDBESSource','GlobalTag')
# process.GlobalTag.globaltag = "COSMMC_21X::All"


process.load("RecoTracker.Configuration.RecoTracker_cff")

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
process.offlineBeamSpot = offlineBeamSpot

from RecoTracker.TrackProducer.TrackRefitters_cff import *
process.CosmicTFRefit = RecoTracker.TrackProducer.TrackRefitterP5_cfi.TrackRefitterP5.clone()
process.CosmicTFRefit.src = 'cosmictrackfinderP5'

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('simul', 
        'cout'),
    simul = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
		"rfio:/castor/cern.ch/cms/store/data/Commissioning08/Cosmics/RECO/CRAFT_V3P_SuperPointing_v4/0039/FEE82D27-23AD-DD11-A657-003048767C73.root"
	),   
#   skipEvents = cms.untracked.uint32(100) 
)

process.lorentzAngle = cms.EDAnalyzer("SiPixelLorentzAngle",
	TTRHBuilder= cms.string("WithTrackAngle"),
	Fitter = cms.string("FittingSmootherRKP5"),
  Propagator = cms.string("RungeKuttaTrackerPropagator"),
#what type of tracks should be used: 
	src = cms.string("CosmicTFRefit"),
	fileName = cms.string("lorentzangle.root"),
	fileNameFit	= cms.string("lorentzFit.txt"),
	binsDepth	= cms.int32(50),
	binsDrift =	cms.int32(200),
	ptMin = cms.double(0),
	#in case of MC set this to true to save the simhits
	simData = cms.bool(False),
  normChi2Max = cms.double(6),	
	clustSizeYMin = cms.int32(3),
	residualMax = cms.double(0.005),
	clustChargeMax = cms.double(120000)
)

process.myout = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('LA_CMSSW.root')
)

process.p = cms.Path(process.offlineBeamSpot*process.CosmicTFRefit*process.lorentzAngle)

# process.outpath = cms.EndPath(process.myout)
