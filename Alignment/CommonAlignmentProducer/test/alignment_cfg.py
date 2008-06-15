# The following comments couldn't be translated into the new config version:

# initialize  MessageLogger

import FWCore.ParameterSet.Config as cms

process = cms.Process("Alignment")
# we need conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

# include "Configuration/StandardSequences/data/FakeConditions.cff"
# initialize magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

# ideal geometry and interface
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

# for Muon: include "Geometry/MuonNumbering/data/muonNumberingInitialization.cfi"
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")

# track selection for alignment
process.load("Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi")

# Alignment producer
process.load("Alignment.CommonAlignmentProducer.AlignmentProducer_cff")

# replace AlignmentProducer.doMisalignmentScenario = true
# replace AlignmentProducer.applyDbAlignment = true # needs other conditions than fake!
# Track refitter (adapted to alignment needs)
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")

process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring('cout', 
        'alignment'),
    categories = cms.untracked.vstring('Alignment'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noLineBreaks = cms.untracked.bool(True)
    ),
    alignment = cms.untracked.PSet(
        INFO = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        noLineBreaks = cms.untracked.bool(True),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        WARNING = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ERROR = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('INFO'),
        Alignment = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        )
    ),
    destinations = cms.untracked.vstring('cout',  ## .log automatically

        'alignment')
)

process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(0),
    fileNames = cms.untracked.vstring('/store/relval/2008/6/4/RelVal-RelValZMM-1212543891-STARTUP-2nd/0000/0A9973E2-9A32-DD11-BE04-001617E30F50.root')
)

process.p = cms.Path(process.offlineBeamSpot+process.AlignmentTrackSelector*process.TrackRefitter)
process.GlobalTag.globaltag = 'IDEAL::All'
process.TrackRefitter.src = 'AlignmentTrackSelector'
process.TrackRefitter.TrajectoryInEvent = True


