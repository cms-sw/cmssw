import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")

# initialize  MessageLogger
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

# we need conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_30X::All'

# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
# for Muon: include "Geometry/MuonNumbering/data/muonNumberingInitialization.cfi"

# track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")

# track refit needs a beamspot in event (irrelevant which one)!:
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# refitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True

# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")
# replace AlignmentProducer.doMisalignmentScenario = true
# replace AlignmentProducer.applyDbAlignment = true

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/4/RelVal-RelValZMM-1212543891-STARTUP-2nd/0000/0A9973E2-9A32-DD11-BE04-001617E30F50.root')
)

process.p = cms.Path(process.offlineBeamSpot+process.AlignmentTrackSelector*process.TrackRefitter)


