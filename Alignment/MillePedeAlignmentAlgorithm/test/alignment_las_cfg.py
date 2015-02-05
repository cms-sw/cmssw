# last update on $Date: 2009/07/17 14:17:33 $ by $Author: kaschube $

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring("ProductNotFound") # do not accept this exception
    )

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
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
#        FwkReport = cms.untracked.PSet(
#            threshold = cms.untracked.string('ERROR')
#        ),
#        TrackProducer = cms.untracked.PSet(
#            threshold = cms.untracked.string('ERROR')
#        )
#    ),
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
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")

# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
# for Muon: process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V3::All' #'IDEAL_V12::All' # 'CRAFT_ALL_V11::All' # 'IDEAL_30X::All'  # take your favourite

# using database file
from CondCore.DBCommon.CondDBSetup_cfi import *
process.trackerAlignment = cms.ESSource("PoolDBESSource",
                                        CondDBSetup,
#                                        connect = cms.string("sqlite_file:/afs/cern.ch/user/k/kaschube/cms/CMSSW_2_2_10/src/LasReader/TestProducer/alignments_MP.db"),
#                                        connect = cms.string("frontier://FrontierProd/CMS_COND_21X_ALIGNMENT"),
                                        connect = cms.string("frontier://FrontierProd/CMS_COND_31X_FROM21X"),
                                        toGet = cms.VPSet(cms.PSet(record = cms.string("TrackerAlignmentRcd"),
                                                                   tag = cms.string("TrackerGeometry_v5_offline")), #"Alignments"
                                                          cms.PSet(record = cms.string("TrackerAlignmentErrorExtendedRcd"),
                                                                   tag = cms.string("TrackerGeometryErrors_v5_offline")) #"AlignmentErrorsExtended"
                                                          )
                                        )
process.es_prefer_trackerAlignment = cms.ESPrefer("PoolDBESSource","trackerAlignment")
#del process.es_prefer_GlobalTag

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'ALCARECOTkAlMinBias' #'generalTracks' ## ALCARECOTkAlMuonIsolated # adjust to input file
process.AlignmentTrackSelector.ptMin = 2.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 9
process.AlignmentTrackSelector.chi2nMax = 100.
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
#         'PixelHalfBarrels,ffffff',
#         'PXEndCaps,111111',
#         'TrackerTOBHalfBarrel,111111',
#         'TrackerTIBHalfBarrel,111111',
#         'TrackerTECEndcap,ffffff',
#         'TrackerTIDEndcap,ffffff'
#
    #'TrackerTPBHalfBarrel,rrrrrr',
    #'TrackerTPEEndcap,rrrrrr',
    #'TrackerTIBHalfBarrel,110111',
    'TrackerTIBHalfBarrel,ff0fff,zIsNeg', # not yet enough fixed d.o.f. ...
    'TrackerTIBHalfBarrel,110111,zIsPos',
    'TrackerTOBHalfBarrel,110111',
    #'TrackerTECEndcap,111111',
    #'TIBSSLayers,110111',
    #'TIBDSLayers,111111',
    #'TOBSSLayers,110111',
    #'TOBDSLayers,111111',
    #'TECLayers,111111'

    'TrackerTECDisk,110001'#,
    #        'TrackerTECEndcap,ff0fff'
    #        'TrackerTECDiskLayers11,ff0fff',
    #        'TrackerTECDiskLayers28,110111',
    #        'TrackerTECDiskLayers99,ff0fff'    
    
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
    ),
    zIsPos = cms.PSet(phiRanges = cms.vdouble(),
                      rRanges = cms.vdouble(),
                      etaRanges = cms.vdouble(),
                      yRanges = cms.vdouble(),
                      xRanges = cms.vdouble(),
                      zRanges = cms.vdouble(-99999., 0.)
                      ),
    zIsNeg = cms.PSet(phiRanges = cms.vdouble(),
                      rRanges = cms.vdouble(),
                      etaRanges = cms.vdouble(),
                      yRanges = cms.vdouble(),
                      xRanges = cms.vdouble(),
                      zRanges = cms.vdouble(0, 99999.)
                      )
)

# LAS first test misalignment
#process.AlignmentProducer.MisalignmentScenario = cms.PSet(
    #saveToDbase = cms.untracked.bool(False),
    #setRotations = cms.bool(True),
    #setTranslations = cms.bool(True),
    #seed = cms.int32(121212),
    #distribution = cms.string('gaussian'),
    #setError = cms.bool(True),

#    TECs = cms.PSet(
#        TECsDisks = cms.PSet(
#            dZlocal = cms.double(0.0),
#            phiXlocal = cms.double(0.0),
#            dYlocal = cms.double(0.05),
#            phiZlocal = cms.double(0.001),
#            dXlocal = cms.double(0.05)#,
#            phiYlocal = cms.double(0.0)
#        )#,
#        TECDisk5 = cms.PSet( # from TrackerNoKnowledgeScenario
#            dZlocal = cms.double(0.0112),
#            phiXlocal = cms.double(0.0002),
#            dYlocal = cms.double(0.01),
#            phiZlocal = cms.double(0.0001),
#            dXlocal = cms.double(0.01)#,
#            phiYlocal = cms.double(0.0001)
#        )   
#        TECDisk2_3_4_5_6_7_8 = cms.PSet(
#            dZlocal = cms.double(0.0112),
#            phiZlocal = cms.double(0.0001),
#            dXlocal = cms.double(0.0112),
#            phiXlocal = cms.double(0.0002),
#            dYlocal = cms.double(0.0206),
#            phiYlocal = cms.double(0.0001)
#        )
#    )
#)

#process.AlignmentProducer.doMuon = True # to align muon system
process.AlignmentProducer.doMisalignmentScenario = True # False
# If the above is true, you might want to choose the scenario:
#from Alignment.TrackerAlignment.Scenarios_cff import *
#process.AlignmentProducer.MisalignmentScenario = TrackerSurveyLASOnlyScenario
process.AlignmentProducer.applyDbAlignment = True # neither globalTag nor trackerAlignment
# monitor not strictly needed:
#process.TFileService = cms.Service("TFileService", fileName = cms.string("histograms.root"))
#process.AlignmentProducer.monitorConfig = cms.PSet(monitors = cms.untracked.vstring ("AlignmentMonitorGeneric"),
#                                                   AlignmentMonitorGeneric = cms.untracked.PSet()
#                                                   )

process.AlignmentProducer.algoConfig = cms.PSet(
    process.MillePedeAlignmentAlgorithm
)

#from Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff import *
#process.AlignmentProducer.algoConfig.pedeSteerer.Presigmas.extend(TrackerShortTermPresigmas.Presigmas)
process.AlignmentProducer.algoConfig.mode = 'full' # 'full' # 'mille' # 'pede' # 'pedeSteer'
process.AlignmentProducer.algoConfig.mergeBinaryFiles = cms.vstring()
process.AlignmentProducer.algoConfig.monitorFile = cms.untracked.string("millePedeMonitor.root")
process.AlignmentProducer.algoConfig.binaryFile = cms.string("milleBinaryISN.dat")
#process.AlignmentProducer.algoConfig.TrajectoryFactory = process.BzeroReferenceTrajectoryFactory
# ...OR TwoBodyDecayTrajectoryFactory OR ...
#process.AlignmentProducer.algoConfig.max2Dcorrelation = 2. # to switch off
#process.AlignmentProducer.algoConfig.fileDir = '/tmp/flucke'
#process.AlignmentProducer.algoConfig.pedeReader.fileDir = './'
#process.AlignmentProducer.algoConfig.treeFile = 'treeFile_lasFirst.root'
##default is sparsGMRES                                    <method>  n(iter)  Delta(F)
process.AlignmentProducer.algoConfig.pedeSteerer.method = 'diagonalisation  6  0.8'
process.AlignmentProducer.algoConfig.pedeSteerer.options = cms.vstring(
    'hugecut 500.0',
   'entries 1' #, 'pedeSteerHierarchy_tecdisks.txt'
   #'chisqcut  20.0  4.5' # ,'outlierdownweighting 3' #,'dwfractioncut 0.1'
   #'bandwidth 6'
)
#process.AlignmentProducer.algoConfig.pedeSteerer.steerFileDebug = True
#process.AlignmentProducer.algoConfig.pedeSteerer.minHieraConstrCoeff = 0.
#process.AlignmentProducer.algoConfig.pedeSteerer.minHieraParPerConstr = 1


process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring(
    #"file:aFile.root" #"rfio:/castor/cern.ch/cms/store/..."
    #"file:/afs/cern.ch/user/k/kaschube/cms/CMSSW_2_2_10/src/Alignment/LaserAlignment/tkLasBeams_dataCRAFT.root"
    "file:/afs/cern.ch/user/f/flucke/cms/CMSSW/CMSSW_3_1_0_pre10/src/tkLasBeams_3_1_X.root"
    # old: tkLasBeams_dataCRAFT.root; new, bad: tkLasBeams_CRAFT_2_2_9.root
    )
)

#process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.load("Alignment.LaserAlignment.TkLasBeamFitter_cfi")
process.AlignmentProducer.tkLasBeamTag = "TkLasBeamFitter"


process.p = cms.Path(process.offlineBeamSpot
                     *process.TkLasBeamFitter
                     #*process.AlignmentTrackSelector
                     #*process.TrackRefitter
                     )

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
#    #  record = cms.string('TrackerAlignmentErrorExtendedRcd'),
#    #  tag = cms.string('testTagAPE') # needed is saveApeToDB = True
#    #)
#                      )
#    )
# MPS needs next line as placeholder for pede _cfg.py:
#MILLEPEDEBLOCK

