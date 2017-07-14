import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load("MagneticField.Engine.volumeBasedMagneticField_160812_cfi")
process.VolumeBasedMagneticFieldESProducer.useParametrizedTrackerField = False

process.testMagneticField = cms.EDAnalyzer("testMagneticField",

    	inputTable = cms.untracked.string("tableList.txt"),
	inputTableType = cms.untracked.string("TOSCAFileList"),
#	inputTableType = cms.untracked.string("xyz_m"),

	resolution     = cms.untracked.double(0.0005), #was: 0.0001
	numberOfPoints = cms.untracked.int32(100000000),

#	InnerRadius = cms.untracked.double(0),    #  default: 0 
#	OuterRadius = cms.untracked.double(900),  #  default: 900 
#       HalfLength  = cms.untracked.double(1600)  #  default: 1600 

)

#process.VolumeBasedMagneticFieldESProducer.gridFiles = cms.VPSet(
          ### Override specs for using specific tables for every volume
#           cms.PSet(
#               volumes   = cms.string('1001-1402,2001-2402'),
#               sectors   = cms.string('0') ,
#               master    = cms.int32(0),
#              path      = cms.string('s[s]/grid.[v].bin'),
#           )
#)


process.p1 = cms.Path(process.testMagneticField)

