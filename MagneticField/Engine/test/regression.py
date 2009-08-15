
import FWCore.ParameterSet.Config as cms

process = cms.Process("MAGNETICFIELDTEST")

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("MagneticField.Engine.volumeBasedMagneticField_090322_2pi_scaled_cfi")

process.testMagneticField = cms.EDAnalyzer("testMagneticField",

## Usine the specified reference file 
	inputTable = cms.untracked.string("/afs/cern.ch/cms/OO/mag_field/CMSSW/regression/referenceField_090322_2pi_scaled.txt"),

## Valid input file types: "xyz_cm", "rpz_m", "xyz_m", "TOSCA" 
	inputTableType = cms.untracked.string("xyz_cm"),

## Resolution used for validation, number of points
	resolution     = cms.untracked.double(0.0001),
	numberOfPoints = cms.untracked.int32(1000000),

## Size of testing volume (cm):
	InnerRadius = cms.untracked.double(0),    #  default: 0 
	OuterRadius = cms.untracked.double(900),  #  default: 900 
        HalfLength  = cms.untracked.double(1600)  #  default: 1600 

)

process.p1 = cms.Path(process.testMagneticField)


