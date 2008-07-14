# last update on $Date: 2008/06/15 07:53:41 $ by $Author: flucke $

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

# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
# for Muon: process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
# process.GlobalTag.globaltag = 'IDEAL_V1::All'

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")
process.AlignmentTrackSelector.src = 'generalTracks' ## ALCARECOTkAlMinBias #cosmictrackfinderP5
process.AlignmentTrackSelector.ptMin = 2.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 9
process.AlignmentTrackSelector.chi2nMax = 100.
process.AlignmentTrackSelector.applyNHighestPt = True
process.AlignmentTrackSelector.nHighestPt = 2

# refitting
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True
# beam halo propagation needs larger phi changes going from one TEC to another
#process.MaterialPropagator.MaxDPhi = 1000.

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
#process.AlignmentProducer.doMuon = True
process.AlignmentProducer.doMisalignmentScenario = False
#process.AlignmentProducer.saveToDB = True

process.AlignmentProducer.algoConfig = cms.PSet(
    process.MillePedeAlignmentAlgorithm
)

from Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff import *
process.AlignmentProducer.algoConfig.pedeSteerer.Presigmas.extend(TrackerShortTermPresigmas.Presigmas)
#process.AlignmentProducer.algoConfig.mode = 'full' # 'mille' # 'pede' # 'pedeSteerer'
#process.AlignmentProducer.algoConfig.TrajectoryFactory = process.BzeroReferenceTrajectoryFactory
# ...OR TwoBodyDecayTrajectoryFactory OR ...
#process.AlignmentProducer.algoConfig.max2Dcorrelation = 2. # to switch off
#process.AlignmentProducer.algoConfig.fileDir = '/tmp/flucke/test'
#process.AlignmentProducer.algoConfig.pedeReader.fileDir = './'
#process.AlignmentProducer.algoConfig.binaryFile = '/tmp/flucke/milleBinary.dat' 
#process.AlignmentProducer.algoConfig.treeFile = 'treeFile_GF.root'
#process.AlignmentProducer.algoConfig.monitorFile = 'millePedeMonitor_GF.root'
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

from CondCore.DBCommon.CondDBSetup_cfi import *
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBSetup,
                                          timetype = cms.untracked.string('runnumber'),
                                          connect = cms.string('sqlite_file:TkAlignment.db'),
                                          toPut = cms.VPSet(cms.PSet(
    record = cms.string('TrackerAlignmentRcd'),
    tag = cms.string('testTag')
    ),
                                                            cms.PSet(
    record = cms.string('TrackerAlignmentErrorRcd'),
    tag = cms.string('testTagAPE')
    ))
                                          )

