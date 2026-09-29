"""
This cfi shows an example of how to activate some debugging tests of the field
map geometry and data tables.

For further information, please refer to
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMagneticField#Development_workflow
"""
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
process.load("MagneticField.Engine.volumeBasedMagneticField_dd4hep_160812_cfi")


### Activate the check of finding volumes at random points.
### This test is useful during developlment of new geometries to check
### that no gaps/holes are present and that all volumes are searchable.
process.testVolumeGeometry = cms.EDAnalyzer("testMagGeometryAnalyzer")
process.p2 = cms.Path(process.testVolumeGeometry) 

### Extra optional checks: Set to True to activate verbose mode for 
# geometry building as well as additional consistency checks on geometry
debugBuilder = False
if (debugBuilder) :
    process.VolumeBasedMagneticFieldESProducer.debugBuilder = True

    # Test grids for all sectors, (also for volumes where tables for sector 1
    # are replicated to reduce memory footprint)
    process.VolumeBasedMagneticFieldESProducer.gridFiles = cms.VPSet(
        cms.PSet(
            volumes   = cms.string('1001-1464,2001-2464'),
            sectors   = cms.string('0') ,
            master    = cms.int32(0),
            path      = cms.string('s[s]/grid.[v].bin'),
        ),
    )

    # Increase verbosity level
    process.MessageLogger = cms.Service("MessageLogger",
        cerr = cms.untracked.PSet(
            enable = cms.untracked.bool(False)
        ),
        cout = cms.untracked.PSet(
            DEBUG = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            ERROR = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            INFO = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            WARNING = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            default = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            enable = cms.untracked.bool(True),
            noLineBreaks = cms.untracked.bool(True),
            threshold = cms.untracked.string('DEBUG')
        ),
        debugModules = cms.untracked.vstring('*')
    )
