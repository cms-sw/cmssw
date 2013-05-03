#

import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Example configuration for the magnetic field

# Uncomment ONE of the following:

### Uniform field
#process.load("Configuration.StandardSequences.MagneticField_0T_cff")
#process.localUniform.ZFieldInTesla = 3.8


### Full field map, static configuration for each field value
#process.load("Configuration.StandardSequences.MagneticField_20T_cff")
#process.load("Configuration.StandardSequences.MagneticField_30T_cff")
#process.load("Configuration.StandardSequences.MagneticField_35T_cff")
#process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_40T_cff")

#process.load("MagneticField.Engine.volumeBasedMagneticField_1103l_cfi")
process.load("MagneticField.Engine.volumeBasedMagneticField_090322_2pi_scaled_cfi")
#process.load("MagneticField.Engine.volumeBasedMagneticField_120812_cfi")

### Configuration to select map based on recorded current in the DB
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'START53_V10::All'
#process.load("MagneticField.Engine.autoMagneticFieldProducer_cfi")
#process.AutoMagneticFieldESProducer.valueOverride = 18000


### Set scaling factors
#process.VolumeBasedMagneticFieldESProducer.scalingVolumes = ( 802 , )
#process.VolumeBasedMagneticFieldESProducer.scalingFactors = ( 1.5 , )


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

