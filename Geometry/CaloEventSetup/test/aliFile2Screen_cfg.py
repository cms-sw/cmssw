import FWCore.ParameterSet.Config as cms

process = cms.Process("read")
process.load('CondCore.CondDB.CondDB_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('EBAlignmentRcd'),
        tag = cms.string('EB')
      ), 
      cms.PSet(
        record = cms.string('EEAlignmentRcd'),
        tag = cms.string('EE')
      ),
      cms.PSet(
        record = cms.string('ESAlignmentRcd'),
        tag = cms.string('ES')
      )
    )
)

##
## Please, rebuild the test with debug enabled
## USER_CXXFLAGS="-g -D=EDM_ML_DEBUG" scram b -v # for bash
## env USER_CXXFLAGS="-g -D=EDM_ML_DEBUG" scram b -v # for tcsh
##
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.cerr.noTimeStamps = cms.untracked.bool(True)
process.MessageLogger.debugModules = cms.untracked.vstring('CaloAlignmentRcdRead')

process.CaloAlignmentRcdRead = cms.EDAnalyzer("CaloAlignmentRcdRead")

process.p = cms.Path(process.CaloAlignmentRcdRead)
