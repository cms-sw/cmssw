import FWCore.ParameterSet.Config as cms
process = cms.Process("MERGE")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
     fileNames = cms.untracked.vstring(
         'FILE1'         
     ),
     secondaryFileNames = cms.untracked.vstring(
         'FILE2'
     ),
     noEventSort = cms.untracked.bool(False),
     duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
 )

process.out = cms.OutputModule("PoolOutputModule",
                                fileName = cms.untracked.string('OUTPUTFILENAME')
                              )
 

process.o = cms.EndPath(process.out)
