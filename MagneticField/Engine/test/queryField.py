# Example configuration for the magnetic field.
# This example prompts for coordinates and prints the corresponding value of B.

import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(300000)
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')


### Standard map, uses current from RunInfo and configuration based on the GT
### (in particular, according to the era)
process.load("Configuration.StandardSequences.MagneticField_cff")

### Override the current from RunInfo with the specified value
#process.VolumeBasedMagneticFieldESProducer.valueOverride = 18000


process.MessageLogger = cms.Service("MessageLogger",
        destinations =  cms.untracked.vstring('cerr'),
        categories = cms.untracked.vstring("MagneticField", # messages on MF configuration, field query
                                           "MagGeoBuilder", # Debug of MF geometry building (debugBuilder flag also needed for full output)
                                           "MagGeometry",   # Debug of MF geometry search                                           
                                           "MagGeometry_cache"), # Volume cache debug
        cerr = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING'), # DEBUG + set limits below
            INFO    = cms.untracked.PSet(limit=cms.untracked.int32(0)),
            DEBUG   = cms.untracked.PSet(limit=cms.untracked.int32(0)),
            WARNING    = cms.untracked.PSet(limit=cms.untracked.int32(0)),
            MagneticField = cms.untracked.PSet(limit=cms.untracked.int32(10000000)),
            MagGeoBuilder = cms.untracked.PSet(limit=cms.untracked.int32(0)),
            MagGeometry = cms.untracked.PSet(limit=cms.untracked.int32(0)),
            MagGeometry_cache = cms.untracked.PSet(limit=cms.untracked.int32(0)),
            ),

        debugModules = cms.untracked.vstring('queryField')
)


process.queryField  = cms.EDAnalyzer("queryField")
process.p1 = cms.Path(process.queryField)

### Activate verbose mode of geometry building as well as additional
### consistency checks on geometry
#process.VolumeBasedMagneticFieldESProducer.debugBuilder = True
