import FWCore.ParameterSet.Config as cms

process = cms.Process("testevtloop")
process.load("FWCore.MessageService.MessageLogger_cfi")

import FWCore.Framework.test.cmsExceptionsFatalOption_cff
process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow = FWCore.Framework.test.cmsExceptionsFatalOption_cff.Rethrow
)
process.maxLuminosityBlocks=cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)

#process.source = cms.Source("EmptySource",
#     numberEventsInRun = cms.untracked.uint32(45),
#     firstRun = cms.untracked.uint32(122314),
#     numberEventsInLuminosityBlock = cms.untracked.uint32(1),
#     firstLuminosityBlock = cms.untracked.uint32(1)
#)

process.source= cms.Source("PoolSource",
              processingMode=cms.untracked.string('RunsAndLumis'),
              fileNames=cms.untracked.vstring(
                    'file:simSample122314-2.root',
                    'file:simSample122314-1.root',
                    'file:simSample122314-3.root',
                    ),             
              firstRun=cms.untracked.uint32(122314),
              #firstLuminosityBlock = cms.untracked.uint32(1),        
              #firstEvent=cms.untracked.uint32(1)
             )
process.test = cms.EDAnalyzer("testEvtLoop")

process.p1 = cms.Path( process.test)

