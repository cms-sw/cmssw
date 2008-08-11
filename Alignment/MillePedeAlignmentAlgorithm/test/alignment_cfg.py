# last update on $Date: 2008/07/15 18:35:21 $ by $Author: flucke $

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")

# initialize  MessageLogger
# process.load("FWCore.MessageLogger.MessageLogger_cfi")
# This whole mess does not really work - I do not get rid of FwkReport and TrackProducer info...
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring('alignment'), ##, 'cout')

    categories = cms.untracked.vstring('Alignment', 
        'LogicError', 
        'FwkReport', 
        'TrackProducer'),
    # FwkReport = cms.untracked.PSet( threshold = cms.untracked.string('WARNING') ),
    # TrackProducer = cms.untracked.PSet( threshold = cms.untracked.string('WARNING') ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        FwkReport = cms.untracked.PSet(
            threshold = cms.untracked.string('ERROR')
        ),
        TrackProducer = cms.untracked.PSet(
            threshold = cms.untracked.string('ERROR')
        )
    ),
    alignment = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(10)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        threshold = cms.untracked.string('DEBUG'),
        LogicError = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
        Alignment = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('alignment') ## (, 'cout')

)

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")
# 0 T field, untested:
#process.load("MagneticField.Engine.uniformMagneticField_cfi")
# Are these prefers still needed?
#process.es_prefer_magfield = cms.ESPrefer("XMLIdealGeometryESSource", "magfield")
#process.es_prefer_uniform = cms.ESPrefer("UniformMagneticFieldESProducer")


# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
# for Muon: process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_V1::All'  # take your favourite
#    # if alignment constants not from global tag, add this
#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.trackerAlignment = cms.ESSource("PoolDBESSource",CondDBSetup,
#                                        connect = cms.string("frontier://FrontierProd/CMS_COND_21X_ALIGNMENT"),
#                                        timetype = cms.string("runnumber"),
#                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
#                                                                   tag = cms.string("TrackerIdealGeometry210_mc")
#                                                                   ),
#                                                          cms.PSet(record = cms.string("TrackerAlignmentErrorRcd"),
#                                                                   tag = cms.string("TrackerIdealGeometryErrors210_mc")
#                                                                   )
#                                                          )
#                                        )
#process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource", "trackerAlignment")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'generalTracks' ## ALCARECOTkAlMinBias # adjust to input file
process.AlignmentTrackSelector.ptMin = 2.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 9
process.AlignmentTrackSelector.chi2nMax = 100.
process.AlignmentTrackSelector.applyNHighestPt = True
process.AlignmentTrackSelector.nHighestPt = 2
# some further possibilities
#process.AlignmentTrackSelector.applyChargeCheck = True
#process.AlignmentTrackSelector.minHitChargeStrip = 50.
# needs RECO files:
#process.AlignmentTrackSelector.applyIsolationCut = True 
#process.AlignmentTrackSelector.minHitIsolation = 0.8


# refitting
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
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
    alignParams = cms.vstring('PixelHalfBarrels,rrrrrr', 
        'TrackerTOBHalfBarrel,111111', 
        'TrackerTIBHalfBarrel,111111', 
        'TrackerTECEndcap,111111', 
        'TrackerTIDEndcap,111111', 
        'PixelDets,111001', 
        'BarrelDetsDS,111001', 
        'TECDets,111001,endCapDS', 
        'TIDDets,111001,endCapDS', 
        'BarrelDetsSS,101001', 
        'TECDets,101001,endCapSS', 
        'TIDDets,101001,endCapSS'
# very simple scenario for testing
#	    # 6 parameters for larger structures, pixel as reference
#        'PixelHalfBarrels,ffffff',
#        'TrackerTOBHalfBarrel,111111',
#        'TrackerTIBHalfBarrel,111111',
#        'TrackerTECEndcap,ffffff',
#        'TrackerTIDEndcap,ffffff' 
                              ),
    endCapSS = cms.PSet(
        phiRanges = cms.vdouble(),
        rRanges = cms.vdouble(40.0, 60.0, 75.0, 999.0),
        etaRanges = cms.vdouble(),
        yRanges = cms.vdouble(),
        xRanges = cms.vdouble(),
        zRanges = cms.vdouble()
    ),
    endCapDS = cms.PSet(
        phiRanges = cms.vdouble(),
        rRanges = cms.vdouble(0.0, 40.0, 60.0, 75.0),
        etaRanges = cms.vdouble(),
        yRanges = cms.vdouble(),
        xRanges = cms.vdouble(),
        zRanges = cms.vdouble()
    )
)
#process.AlignmentProducer.doMuon = True # to align muon system
process.AlignmentProducer.doMisalignmentScenario = False
# If the above is true, you might want to choose the scenario:
#from Alignment.TrackerAlignment.Scenarios_cff import *
#process.AlignmentProducer.MisalignmentScenario = TrackerSurveyLASOnlyScenario
process.AlignmentProducer.applyDbAlignment = True #false # otherwise neither globalTag not trackerAlignment
# monitor not strictly needed:
#process.TFileService = cms.Service("TFileService", fileName = cms.string("histograms.root"))
#process.AlignmentProducer.monitorConfig = cms.PSet(monitors = cms.untracked.vstring ("AlignmentMonitorGeneric"),
#                                                   AlignmentMonitorGeneric = cms.untracked.PSet()
#                                                   )

process.AlignmentProducer.algoConfig = cms.PSet(
    process.MillePedeAlignmentAlgorithm
)

from Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff import *
process.AlignmentProducer.algoConfig.pedeSteerer.Presigmas.extend(TrackerShortTermPresigmas.Presigmas)
process.AlignmentProducer.algoConfig.mode = 'full' # 'mille' # 'pede' # 'pedeSteerer'
process.AlignmentProducer.algoConfig.mergeBinaryFiles = cms.vstring()
process.AlignmentProducer.algoConfig.monitorFile = cms.untracked.string("millePedeMonitor.root")
process.AlignmentProducer.algoConfig.binaryFile = cms.string("milleBinaryISN.dat")
#process.AlignmentProducer.algoConfig.TrajectoryFactory = process.BzeroReferenceTrajectoryFactory
# ...OR TwoBodyDecayTrajectoryFactory OR ...
#process.AlignmentProducer.algoConfig.max2Dcorrelation = 2. # to switch off
#process.AlignmentProducer.algoConfig.fileDir = '/tmp/flucke/test'
#process.AlignmentProducer.algoConfig.pedeReader.fileDir = './'
#process.AlignmentProducer.algoConfig.treeFile = 'treeFile_GF.root'
##default is sparsGMRES                                    <method>  n(iter)  Delta(F)
#process.AlignmentProducer.algoConfig.pedeSteerer.method = 'inversion  9  0.8'
#process.AlignmentProducer.algoConfig.pedeSteerer.options = cms.vstring(
#   'entries 100',
#   'chisqcut  20.0  4.5' # ,'outlierdownweighting 3' #,'dwfractioncut 0.1' 
#)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValZMM-1213987236-IDEAL_V2-2nd/0004/04666D76-1941-DD11-9549-001617E30E28.root'
                                      # <== is a relval file from CMSSW_2_1_0_pre8.
                                      #"file:aFile.root" #"rfio:/castor/cern.ch/cms/store/..."
                                      )
)
#process.source = cms.Source("EmptySource")
#process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(0)
#    )

process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackRefitter)

#--- SAVE ALIGNMENT CONSTANTS TO DB --------------------------------
# Default in MPS is saving as alignment_MP.db. Uncomment next line not to save them.
# For a standalone (non-MPS) run, uncomment also the PoolDBOutputService part.
#process.AlignmentProducer.saveToDB = True
#from CondCore.DBCommon.CondDBSetup_cfi import *
#process.PoolDBOutputService = cms.Service("PoolDBOutputService",
#                                          CondDBSetup,
#                                          timetype = cms.untracked.string('runnumber'),
#                                          connect = cms.string('sqlite_file:TkAlignment.db'),
#                                          toPut = cms.VPSet(cms.PSet(
#    record = cms.string('TrackerAlignmentRcd'),
#    tag = cms.string('testTag')
#    ),
#                                                            cms.PSet(
#    record = cms.string('TrackerAlignmentErrorRcd'),
#    tag = cms.string('testTagAPE')
#    ))
#                                          )
# MPS needs next line as placeholder for pede _cfg.py:
#MILLEPEDEBLOCK

