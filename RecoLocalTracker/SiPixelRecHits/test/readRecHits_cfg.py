#
# rechits are not persisent anymore, so one should run one of the CPEs
# on clusters ot do the track fitting. 11/08 d.k.
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("recHitsTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('ReadPixelRecHit'),
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
   fileNames =  cms.untracked.vstring('file:/scratch/dkotlins/promptrecoCosmics_1.root')
)


# a service to use root histos
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
process.load("Configuration.StandardSequences.FakeConditions_cff")

# Initialize magnetic field
# include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
# Tracker SimGeometryXML
# include "Geometry/TrackerSimData/data/trackerSimGeometryXML.cfi"
# Tracker Geometry Builder
# include "Geometry/TrackerGeometryBuilder/data/trackerGeometry.cfi" 
# Tracker Numbering Builder
# include "Geometry/TrackerNumberingBuilder/data/trackerNumberingGeometry.cfi"

process.analysis = cms.EDAnalyzer("ReadPixelRecHit",
    Verbosity = cms.untracked.bool(True),
    src = cms.InputTag("siPixelRecHits"),
)

process.p = cms.Path(process.analysis)





