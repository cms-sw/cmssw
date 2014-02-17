# last update on $Date: 2010/02/26 12:14:59 $ by $Author: frmeier $

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound") # do not accept this exception
    )

# initialize  MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.alignment = cms.untracked.PSet(
    DEBUG = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        ),
    INFO = cms.untracked.PSet(
        limit = cms.untracked.int32(5),
        reportEvery = cms.untracked.int32(5)
        ),
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(10)
        ),
    ERROR = cms.untracked.PSet(
        limit = cms.untracked.int32(-1)
        ),
    Alignment = cms.untracked.PSet(
        limit = cms.untracked.int32(-1),
        reportEvery = cms.untracked.int32(1)
        )
    )
process.MessageLogger.cerr.placeholder = cms.untracked.bool(True)
process.MessageLogger.destinations = ['alignment']
process.MessageLogger.statistics = ['alignment']
process.MessageLogger.categories = ['Alignment']


# initialize magnetic field
#process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")

# geometry
process.load("Configuration.StandardSequences.Geometry_cff")
#del process.CaloTopologyBuilder etc. to speed up...???

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'MC_3XY_V14::All' # take your favourite
process.GlobalTag.globaltag = 'CRAFT09_R_V10::All' # take your favourite
#    # if alignment constants not from global tag, add this
#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.trackerAlignment = cms.ESSource(
#    "PoolDBESSource",
#    CondDBSetup,
#    connect = cms.string("frontier://FrontierProd/CMS_COND_31X_ALIGNMENT"),
##    connect = cms.string("frontier://FrontierPrep/CMS_COND_ALIGNMENT"),
#    toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
#                               tag = cms.string("TrackerIdealGeometry210_mc")
#                               ),
#                      cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
#                               tag = cms.string("TrackerIdealGeometryErrors210_mc")
#                               )
#                      )
#    )
#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")
## might help for double es_prefer: del process.es_prefer_GlobalTag

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'ALCARECOTkAlZMuMu' #MinBias' #'generalTracks' ## ALCARECOTkAlMuonIsolated # adjust to input file
process.AlignmentTrackSelector.ptMin = 8.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 9
process.AlignmentTrackSelector.chi2nMax = 50.
# some further possibilities
#process.AlignmentTrackSelector.applyNHighestPt = True
#process.AlignmentTrackSelector.nHighestPt = 2
#process.AlignmentTrackSelector.applyChargeCheck = True
#process.AlignmentTrackSelector.minHitChargeStrip = 50.
# needs RECO files:
#process.AlignmentTrackSelector.applyIsolationCut = True 
#process.AlignmentTrackSelector.minHitIsolation = 0.8


# refitting
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
# In the following use
# TrackRefitter (normal tracks), TrackRefitterP5 (cosmics) or TrackRefitterBHM (beam halo)
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True
# beam halo propagation needs larger phi changes going from one TEC to another
#process.MaterialPropagator.MaxDPhi = 1000.
# the following for refitting with analytical propagator (maybe for CRUZET?)
#process.load("TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi")
#process.load("TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi")
#process.load("TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi")
#process.load("TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi")
#process.load("TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi")
#process.load("TrackingTools.GeomPropagators.AnalyticalPropagator_cfi")
#process.load("TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi")
#process.TrackRefitter.Propagator = "AnalyticalPropagator"
#process.KFTrajectoryFitter.Propagator = "AnalyticalPropagator"
#process.KFTrajectorySmoother.Propagator = "AnalyticalPropagator"
## Not to loose hits/tracks, we might want to open the allowed chi^2 contribution for single hits:
##process.Chi2MeasurementEstimator.MaxChi2 = 50. # untested, default 30
#process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
#process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi")
#process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi")
#process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi")
## end refitting with analytical propagator


# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")

process.AlignmentProducer.ParameterBuilder.Selector = cms.PSet(
    alignParams = cms.vstring(
#        'TrackerTECPetalLayers11,fff00f', #
#        'TrackerTECPetalLayers29,111001', #
#        'TrackerTOBRodLayers35,101001', #
#        'TrackerTOBRodLayers66,f0f00f', #
#        'TrackerTOBRodDS,111001', #
#        'TrackerTIDRing,111001', #
#        'TrackerTIBStringSS,101001',#
#        'TrackerTIBStringDS,111001',#
#        'TrackerTPEHalfDisk,111111',
#        'TrackerTPBLadderLayers11,ffff0f', #
#        'TrackerTPBLadderLayers23,111101'  #
#
#        'PixelHalfBarrels,rrrrrr', 
#        'PXEndCaps,111111',
#        'TrackerTOBHalfBarrel,111111', 
#        'TrackerTIBHalfBarrel,111111', 
#        'TrackerTECEndcap,111111', 
#        'TrackerTIDEndcap,111111', 
#        'PixelDets,111001', 
#        'BarrelDetsDS,111001', 
#        'TECDets,111001,endCapDS', 
#        'TIDDets,111001,endCapDS', 
#        'BarrelDetsSS,101001', 
#        'TECDets,101001,endCapSS', 
#        'TIDDets,101001,endCapSS'
#
# very simple scenario for testing
# # 6 parameters for larger structures
         'PixelHalfBarrels,ffffff',
         'PXEndCaps,111111',
         'TrackerTOBHalfBarrel,111111',
         'TrackerTIBHalfBarrel,111111',
         'TrackerTECEndcap,ffffff',
         'TrackerTIDEndcap,ffffff' 
         ),
    endCapSS = cms.PSet(
        rRanges = cms.vdouble(40.0, 60.0, 75.0, 999.0)
    ),
    endCapDS = cms.PSet(
        rRanges = cms.vdouble(0.0, 40.0, 60.0, 75.0)
    )
)
#process.AlignmentProducer.doMuon = True # to align muon system
process.AlignmentProducer.doMisalignmentScenario = False #True
# If the above is true, you might want to choose the scenario:
#from Alignment.TrackerAlignment.Scenarios_cff import *
#process.AlignmentProducer.MisalignmentScenario = TrackerSurveyLASOnlyScenario
process.AlignmentProducer.applyDbAlignment = True # neither globalTag nor trackerAlignment

# assign by reference (i.e. could change MillePedeAlignmentAlgorithm as well):
process.AlignmentProducer.algoConfig = process.MillePedeAlignmentAlgorithm

#from Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff import *
#process.AlignmentProducer.algoConfig.pedeSteerer.Presigmas.extend(TrackerShortTermPresigmas.Presigmas)
process.AlignmentProducer.algoConfig.mode = 'full' # 'mille' # 'pede' # 'pedeSteerer'
process.AlignmentProducer.algoConfig.binaryFile = 'milleBinaryISN.dat'
process.AlignmentProducer.algoConfig.treeFile = 'treeFileISN.root'

#process.AlignmentProducer.algoConfig.TrajectoryFactory = process.TwoBodyDecayTrajectoryFactory # BzeroReferenceTrajectoryFactory
#process.AlignmentProducer.algoConfig.TrajectoryFactory.PropagationDirection = 'oppositeToMomentum'
process.AlignmentProducer.algoConfig.TrajectoryFactory.MaterialEffects = 'BrokenLinesCoarse' #'BreakPoints' #'Combined', ## or "MultipleScattering" or "EnergyLoss" or "None"
# ...OR TwoBodyDecayTrajectoryFactory OR ...
#process.AlignmentProducer.algoConfig.max2Dcorrelation = 2. # to switch off
##default is sparsGMRES                                    <method>  n(iter)  Delta(F)
#process.AlignmentProducer.algoConfig.pedeSteerer.method = 'inversion  9  0.8'
#process.AlignmentProducer.algoConfig.pedeSteerer.options = cms.vstring(
#   'entries 100',
#   'chisqcut  20.0  4.5' # ,'outlierdownweighting 3' #,'dwfractioncut 0.1'
#)
#process.AlignmentProducer.algoConfig.pedeSteerer.pedeCommand = '/afs/cern.ch/user/f/flucke/cms/pede/MillepedeII/pede_1GB'

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(0) )

#process.source = cms.Source("PoolSource",
#    skipEvents = cms.untracked.uint32(0),
#    fileNames = cms.untracked.vstring(
#        #'/store/relval/CMSSW_3_0_0_pre2/RelValTTbar/ALCARECO/STARTUP_V7_StreamALCARECOTkAlMinBias_v2/0001/580E7A0F-1DB4-DD11-8AA8-001617DBD224.root'
#        # <== is a relval from CMSSW_3_0_0_pre2.
#        #"file:aFile.root" #"rfio:/castor/cern.ch/cms/store/..."
#        #'/store/mc/Summer09/MinBias/ALCARECO/MC_31X_V3_StreamTkAlMinBias-v2/0005/C2F77DDA-0292-DE11-BD6B-001731A2870D.root'
#        # <== from /MinBias/Summer09-MC_31X_V3_StreamTkAlMinBias-v2/ALCARECO
#        '/store/mc/Summer09/Zmumu/ALCARECO/MC_31X_V3_7TeV_StreamTkAlZMuMu-v1/0000/FACBA5D7-B1A2-DE11-AE4A-001E4F3F28D8.root'
#        # <== /Zmumu/Summer09-MC_31X_V3_7TeV_StreamTkAlZMuMu-v1/ALCARECO
#        #'file:/tmp/flucke/Summer09_Zmumu_ALCARECO_MC_31X_V3_7TeV_StreamTkAlZMuMu-v1_0000_FACBA5D7-B1A2-DE11-AE4A-001E4F3F28D8.root'
#
#    )
#                            )
#process.source = cms.Source("EmptySource")
#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(0)
#    )

process.AlignmentProducer.algoConfig.surveyPixelBarrel.doSurvey = True
process.AlignmentProducer.algoConfig.surveyPixelBarrel.doOutputOnStdout = True
process.AlignmentProducer.algoConfig.surveyPixelBarrel.infile = cms.FileInPath("Alignment/MillePedeAlignmentAlgorithm/test/toySurveyInfo1.txt")
process.AlignmentProducer.algoConfig.surveyPixelBarrel.doToySurvey = True
process.AlignmentProducer.algoConfig.surveyPixelBarrel.toySurveyParameters = cms.VPSet(
		# Position of photo in local frame (unit: pixels in photo)
		cms.PSet(name = cms.string('a0'),    mean = cms.double(1879.11,), sigma = cms.double(0.)),
		cms.PSet(name = cms.string('a1'),    mean = cms.double(2476.56,), sigma = cms.double(0.)),
		# Scale of photo (unit: pixels per cm)
		cms.PSet(name = cms.string('scale'), mean = cms.double(1168.68), sigma = cms.double(0.)),
		# Rotation of photo in local frame (unit: rads)
		cms.PSet(name = cms.string('phi'),   mean = cms.double(0.00887099), sigma = cms.double(0.)),
		# Smearing of measurements in u and v coordinate (unit: pixels in photo)
		cms.PSet(name = cms.string('u'),     mean = cms.double(0.), sigma = cms.double(0.175)),
		cms.PSet(name = cms.string('v'),     mean = cms.double(0.), sigma = cms.double(0.175))
		)


process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackRefitter)

# all fits/refits with 'StripCPEgeometric' - but take about TTRHBuilder used ibn (re)fit
# (works until 'StripCPEgeometric' gets default...)
#process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEgeometric_cfi")
#process.TTRHBuilderAngleAndTemplate.StripCPE = 'StripCPEgeometric'

#--- SAVE ALIGNMENT CONSTANTS TO DB --------------------------------
# Default in MPS is saving as alignment_MP.db. Uncomment next line not to save them.
# For a standalone (non-MPS) run, uncomment also the PoolDBOutputService part.
#process.AlignmentProducer.saveToDB = True
##process.AlignmentProducer.saveApeToDB = True # no sense: Millepede does not set it!
#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.PoolDBOutputService = cms.Service(
#    "PoolDBOutputService",
#    CondDBSetup,
#    timetype = cms.untracked.string('runnumber'),
#    connect = cms.string('sqlite_file:TkAlignment.db'),
#    toPut = cms.VPSet(cms.PSet(
#      record = cms.string('TrackerAlignmentRcd'),
#      tag = cms.string('testTag')
#    )#,
#    #                  cms.PSet(
#    #  record = cms.string('TrackerAlignmentErrorRcd'),
#    #  tag = cms.string('testTagAPE') # needed is saveApeToDB = True
#    #)
#                      )
#    )
# MPS needs next line as placeholder for pede _cfg.py:
#MILLEPEDEBLOCK

