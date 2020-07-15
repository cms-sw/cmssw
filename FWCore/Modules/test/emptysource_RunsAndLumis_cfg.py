import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.source = cms.Source("EmptySource", 
                            numberEventsInLuminosityBlock = cms.untracked.uint32(2), 
                            processingMode=cms.untracked.string('RunsAndLumis'))

process.maxLuminosityBlocks.input = 3

ids = cms.untracked.VPSet()
ids.append(cms.PSet(type = cms.untracked.string("IsFile"),
                    id = cms.untracked.EventID(0,0,0)))
for i in range(0,4):
    if(i==0) :
        ids.append(cms.PSet(type = cms.untracked.string("IsRun"),
                                id = cms.untracked.EventID(1,0,0)))
    else:
        ids.append(cms.PSet(type = cms.untracked.string("IsLumi"),
                                id = cms.untracked.EventID(1,i,0)))

ids.append(cms.PSet(type = cms.untracked.string("IsStop"),
                    id = cms.untracked.EventID(0,0,0)))

#print ids.dumpPython()
process.add_(cms.Service("CheckTransitions",
                         transitions = ids))

