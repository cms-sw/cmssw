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

### Configuration to select map based on recorded current in the DB
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag,'auto:phase1_2017_realistic', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load("MagneticField.Engine.autoMagneticFieldProducer_cfi")
process.AutoMagneticFieldESProducer.valueOverride = 18000


process.myTest  = cms.EDAnalyzer("MSGalore",
 measurementTracker = cms.string(''),
 navigationSchool   = cms.string('SimpleNavigationSchool')
)
process.p1 = cms.Path(process.myTest)

