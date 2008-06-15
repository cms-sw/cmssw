# The following comments couldn't be translated into the new config version:

# last update on $Date: 2008/06/13 12:43:34 $ by $Author: flucke $

# initialize  MessageLogger
#    include "FWCore/MessageLogger/data/MessageLogger.cfi"
# This whole mess does not really work - I do not get rid of FwkReport and TrackProducer info...

# 999999.
import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")
# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# for Muon: include "Geometry/MuonNumbering/data/muonNumberingInitialization.cfi"
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")

# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")

#    replace MillePedeAlignmentAlgorithm.mode = "mille" #"full" # "pedeSteer" #"pede"
#    replace MillePedeAlignmentAlgorithm.fileDir = "/scratch/flucke/lxbatch/milleCosmics"
#    replace MillePedeAlignmentAlgorithm.binaryFile = "/tmp/flucke/milleBinary.dat" 
#    replace MillePedeAlignmentAlgorithm.treeFile = "treeFile_hiera131d.root"
#    replace MillePedeAlignmentAlgorithm.monitorFile = "millePedeMonitor_hiera131d.root"
#    default is sparsGMRES                                    <method>  n(iter)  Delta(F)
#    replace MillePedeAlignmentAlgorithm.pedeSteerer.method = "inversion  9  0.8"
#    replace MillePedeAlignmentAlgorithm.pedeSteerer.options = {
#	"entries 100",
#	"chisqcut  20.0  4.5" # "outlierdownweighting 3", "dwfractioncut 0.1" 
#    }
#    replace MillePedeAlignmentAlgorithm.TrajectoryFactory = {
#      using BzeroReferenceTrajectoryFactory # OR using TwoBodyDecayTrajectoryFactory OR using...
#    }
# replace MillePedeAlignmentAlgorithm.max2Dcorrelation = 2. # to switch off
# FIXME: Need updated presigma scenarios (or implement use of survey object?)
process.load("Alignment.MillePedeAlignmentAlgorithm.PresigmaScenarios_cff")

# refitting
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring('alignment'), ##{, "cout" }

    categories = cms.untracked.vstring('Alignment', 
        'LogicError', 
        'FwkReport', 
        'TrackProducer'),
    #untracked PSet FwkReport     = { untracked string threshold = "WARNING" }
    #untracked PSet TrackProducer = { untracked string threshold = "WARNING" }
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
    destinations = cms.untracked.vstring('alignment') ## {, "cout" }

)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/4/RelVal-RelValZMM-1212543891-STARTUP-2nd/0000/0A9973E2-9A32-DD11-BE04-001617E30F50.root')
)

process.p = cms.Path(process.offlineBeamSpot*process.AlignmentTrackSelector*process.TrackRefitter)
process.GlobalTag.globaltag = 'IDEAL::All'
process.AlignmentTrackSelector.src = 'generalTracks'
process.AlignmentTrackSelector.ptMin = 2.
process.AlignmentTrackSelector.etaMin = -5.
process.AlignmentTrackSelector.etaMax = 5.
process.AlignmentTrackSelector.nHitMin = 9
process.AlignmentTrackSelector.chi2nMax = 100.
process.AlignmentTrackSelector.applyNHighestPt = True
process.AlignmentTrackSelector.nHighestPt = 2
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
        'TIDDets,101001,endCapSS'),
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
process.AlignmentProducer.doMisalignmentScenario = False
process.MillePedeAlignmentAlgorithm.pedeSteerer.Presigmas.extend(process.TrackerShortTermPresigmas.Presigmas)
process.AlignmentProducer.algoConfig = cms.PSet(
    process.MillePedeAlignmentAlgorithm
)
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True


