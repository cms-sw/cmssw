import FWCore.ParameterSet.Config as cms

#Run = 173663
#Run = 175045
#Run = 173692
Run = 172822
process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:test_21_a_1_RAW2DIGI_RECO_DQM.root"),
                            filterOnRun = cms.untracked.uint32(Run))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("test_21_a_1_"+str(Run)+"_RAW2DIGI_RECO_DQM.root"),
                               filterOnRun = cms.untracked.uint32(Run))
process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore"))
