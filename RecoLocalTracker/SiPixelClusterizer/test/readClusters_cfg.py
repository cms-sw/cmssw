#
# Last update: new version for python
#
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("cluTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelClusters'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
#    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/digis.root')
    fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/COSMIC/RECO/005102D1-ACD5-DE11-AD13-000423D98BC4.root')
)


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# what is this?
# process.load("Configuration.StandardSequences.Services_cff")

# what is this?
#process.load("SimTracker.Configuration.SimTracker_cff")

# needed for global transformation
# process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
#process.GlobalTag.globaltag = 'MC_31X_V9::All'
process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'

# Initialize magnetic field
#  include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
# Tracker SimGeometryXML
#  include "Geometry/TrackerSimData/data/trackerSimGeometryXML.cfi"
# Tracker Geometry Builder
#  include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi"
# Tracker Numbering Builder
#  include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"
 
process.analysis = cms.EDAnalyzer("ReadPixClusters",
#    VerboseLevel = cms.untracked.int32(1),
    src = cms.InputTag("siPixelClusters"),
)

process.p = cms.Path(process.analysis)



