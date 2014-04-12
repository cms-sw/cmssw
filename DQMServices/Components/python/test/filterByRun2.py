import FWCore.ParameterSet.Config as cms
import sys

#Run = 173663
#Run = 175045
#Run = 173692
#Run = 172791
Run=sys.argv[2]
tnum=41

print 'Filtering out Run %s' % Run

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:file_"+str(tnum)+"_a_new_Run172791_173241_173243_173244.root"),
                            filterOnRun = cms.untracked.uint32(int(Run)))

process.out = cms.OutputModule("DQMRootOutputModule",
                               fileName = cms.untracked.string("file_"+str(tnum)+"_a_new_Run" + Run + ".root"),
                               filterOnRun = cms.untracked.uint32(int(Run)))
process.e = cms.EndPath(process.out)

process.add_(cms.Service("DQMStore"))
