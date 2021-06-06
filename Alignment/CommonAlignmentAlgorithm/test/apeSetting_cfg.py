import FWCore.ParameterSet.Config as cms

process = cms.Process("APE")
# we need conditions
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_V11::All'

# include "Configuration/StandardSequences/data/FakeConditions.cff"
# initialize magnetic field
#process.load("Configuration.StandardSequences.MagneticField_cff")

# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
# for Muon: include "Geometry/MuonNumbering/data/muonNumberingInitialization.cfi"

# track selection for alignment
#process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")

# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
from Alignment.CommonAlignmentAlgorithm.ApeSettingAlgorithm_cfi import *
process.AlignmentProducer.algoConfig = ApeSettingAlgorithm
process.AlignmentProducer.saveApeToDB = True
process.AlignmentProducer.algoConfig.readApeFromASCII = True
process.AlignmentProducer.algoConfig.setComposites = False
process.AlignmentProducer.algoConfig.readLocalNotGlobal = False
process.AlignmentProducer.algoConfig.apeASCIIReadFile = 'Alignment/CommonAlignmentAlgorithm/test/ApeDump.txt'
process.AlignmentProducer.algoConfig.saveApeToASCII = True
process.AlignmentProducer.algoConfig.saveComposites = False
process.AlignmentProducer.algoConfig.apeASCIISaveFile = 'myDump.txt'
        
# replace AlignmentProducer.doMisalignmentScenario = true
# replace AlignmentProducer.applyDbAlignment = true # needs other conditions than fake!
# Track refitter (adapted to alignment needs)
#process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

# to be refined...
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        enableStatistics = cms.untracked.bool(True),
        noLineBreaks = cms.untracked.bool(True),
        threshold = cms.untracked.string('DEBUG')
    ),
    files = cms.untracked.PSet(
        alignment = cms.untracked.PSet(
            Alignment = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            enableStatistics = cms.untracked.bool(True),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(0) )

from CondCore.DBCommon.CondDBSetup_cfi import *
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondDBSetup,
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:TkAlignmentApe.db'),
    toPut = cms.VPSet(cms.PSet(record = cms.string('TrackerAlignmentErrorRcd'),
                               tag = cms.string('testTagAPE')
                               )
                      )
    )



# We do not even need a path - producer is called anyway...
#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
#process.p = cms.Path(process.offlineBeamSpot)
#process.TrackRefitter.src = 'AlignmentTrackSelector'
#process.TrackRefitter.TrajectoryInEvent = True


