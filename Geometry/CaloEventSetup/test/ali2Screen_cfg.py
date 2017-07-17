import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.load('Geometry.CaloEventSetup.FakeCaloAlignments_cff')

process.CaloAlignmentRcdRead = cms.EDAnalyzer("CaloAlignmentRcdRead")

##
## Please, rebuild the test with debug enabled
## USER_CXXFLAGS="-g -D=EDM_ML_DEBUG" scram b -v # for bash
## env USER_CXXFLAGS="-g -D=EDM_ML_DEBUG" scram b -v # for tcsh
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.noTimeStamps = cms.untracked.bool(True)
process.MessageLogger.debugModules = cms.untracked.vstring('CaloAlignmentRcdRead')

process.p = cms.Path(process.CaloAlignmentRcdRead)
