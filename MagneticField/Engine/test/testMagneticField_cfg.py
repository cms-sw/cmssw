# This cfi shows an example of how to activate some debugging tests of the field
# map geometry and data tables.
# For further information, please refer to
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMagneticField#Development_workflow

import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(300000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


### Static configuration for a given field map
### Please note that except for DEBUGGING you should use the standard sequence
### Configuration.StandardSequences.MagneticField_cff instead!
process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")


process.MessageLogger = cms.Service("MessageLogger",
    categories   = cms.untracked.vstring("MagneticField"),
    destinations = cms.untracked.vstring("cout"),
    cout = cms.untracked.PSet(  
    noLineBreaks = cms.untracked.bool(True),
    threshold = cms.untracked.string("WARNING"),
    WARNING = cms.untracked.PSet(
      limit = cms.untracked.int32(0)
    ),
    MagneticField = cms.untracked.PSet(
     limit = cms.untracked.int32(1)
    )
  )
)

process.testField  = cms.EDAnalyzer("testMagneticField")
process.p1 = cms.Path(process.testField)


### Activate the check of finding volumes at random points.
### This test is useful during developlment of new geometries to check
### that no gaps/holes are present and that all volumes are searchable.
process.testVolumeGeometry = cms.EDAnalyzer("testMagGeometryAnalyzer")
process.p2 = cms.Path(process.testVolumeGeometry) 

### Activate verbose mode of geometry building as well as additional
### consistency checks on geometry
#process.VolumeBasedMagneticFieldESProducer.debugBuilder = True

