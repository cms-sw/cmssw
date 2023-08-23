from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import sys

def chooseTrans(index):
    d = (
         ["Simple", #0
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,4)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,5)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Less events than streams", #1
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Multiple different Lumis", #2
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,2,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,4)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Empty Lumi", #3
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,2,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,4)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Empty Lumi at end", #4
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,4)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,2,0)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Multiple different runs", #5
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(2,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(2,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,2)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Empty run", #6
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(2,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(2,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,2)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Empty run at end", #7
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(2,0,0)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Empty run no lumi", #8
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Empty file at end", #9
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Empty file", #10
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Merge run across files", #11
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,2,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,4)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Merge run & lumi across files", #12
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,4)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Different run across files", #13
          cms.untracked.VPSet(
                              cms.PSet(type = cms.untracked.string("IsFile"),
                                       id = cms.untracked.EventID(0,0,0)),
                              cms.PSet(type = cms.untracked.string("IsRun"),
                                       id = cms.untracked.EventID(1,0,0)),
                              cms.PSet(type = cms.untracked.string("IsLumi"),
                                       id = cms.untracked.EventID(1,1,0)),
                              cms.PSet(type = cms.untracked.string("IsEvent"),
                                       id = cms.untracked.EventID(1,1,1)),
                              cms.PSet(type = cms.untracked.string("IsEvent"),
                                       id = cms.untracked.EventID(1,1,2)),
                              cms.PSet(type = cms.untracked.string("IsFile"),
                                       id = cms.untracked.EventID(0,0,0)),
                              cms.PSet(type = cms.untracked.string("IsRun"),
                                       id = cms.untracked.EventID(2,0,0)),
                              cms.PSet(type = cms.untracked.string("IsLumi"),
                                       id = cms.untracked.EventID(2,1,0)),
                              cms.PSet(type = cms.untracked.string("IsEvent"),
                                       id = cms.untracked.EventID(2,1,1)),
                              cms.PSet(type = cms.untracked.string("IsEvent"),
                                       id = cms.untracked.EventID(2,1,2)),
                              cms.PSet(type = cms.untracked.string("IsStop"),
                                       id = cms.untracked.EventID(0,0,0))
                              ) ],

         ["Delayed lumi merge", #14
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"), #to merge
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,2,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,4)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Delayed lumi merge 2", #15
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,2,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,2,0)), # to merge
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,3)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,2,4)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Delayed run merge", #16
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"), # to merge
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(2,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(2,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,2)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
         ["Delayed run merge 2", #17
    cms.untracked.VPSet(
                        cms.PSet(type = cms.untracked.string("IsFile"),
                                 id = cms.untracked.EventID(0,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(1,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(1,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(1,1,2)),
                        cms.PSet(type = cms.untracked.string("IsRun"),
                                 id = cms.untracked.EventID(2,0,0)),
                        cms.PSet(type = cms.untracked.string("IsRun"), # to merge
                                 id = cms.untracked.EventID(2,0,0)),
                        cms.PSet(type = cms.untracked.string("IsLumi"),
                                 id = cms.untracked.EventID(2,1,0)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,1)),
                        cms.PSet(type = cms.untracked.string("IsEvent"),
                                 id = cms.untracked.EventID(2,1,2)),
                        cms.PSet(type = cms.untracked.string("IsStop"),
                                 id = cms.untracked.EventID(0,0,0))
                        ) ],
    )
    print('****************************************')
    print('Test:', d[index][0])
    print('****************************************')
    return d[index][1]

trans = chooseTrans(int(sys.argv[1]))

process = cms.Process("TEST")
process.source = cms.Source("TestSource", 
                            transitions = trans)
#process.add_(cms.Service("Tracer"))
process.add_(cms.Service("CheckTransitions", 
                         transitions = trans))
process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(2),
                                     numberOfStreams = cms.untracked.uint32(0))
