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


# process.MessageLogger = cms.Service("MessageLogger",
#     categories   = cms.untracked.vstring("MagneticField"),
#     destinations = cms.untracked.vstring("cout"),
#     cout = cms.untracked.PSet(  
#     noLineBreaks = cms.untracked.bool(True),
#     threshold = cms.untracked.string("INFO"),
#     INFO = cms.untracked.PSet(
#       limit = cms.untracked.int32(0)
#     ),
#     WARNING = cms.untracked.PSet(
#       limit = cms.untracked.int32(0)
#     ),
#     MagneticField = cms.untracked.PSet(
#      limit = cms.untracked.int32(10000000)
#     )
#   )
# )

process.queryField  = cms.EDAnalyzer("queryField")
process.p1 = cms.Path(process.queryField)

