import FWCore.ParameterSet.Config as cms
from FWCore.Framework.modules import AddIntsProducer, IntProductFilter
from FWCore.Modules.modules import AsciiOutputModule
from FWCore.Integration.modules import DelayedReaderThrowingSource

process = cms.Process("TEST")

process.source = DelayedReaderThrowingSource( labels = ["test", "test2", "test3"])

process.getter = AddIntsProducer(labels = [("test","","INPUTTEST")])
process.onPath = AddIntsProducer(labels = [("test2", "", "INPUTTEST"), ("getter", "other")])
process.f1 = IntProductFilter( label = "onPath", shouldProduce = True)
process.f2 = IntProductFilter( label = "onPath", shouldProduce = True)
process.inFront = IntProductFilter( label = "test3")

process.p1 = cms.Path(process.inFront+process.onPath+process.f1+process.f2)
process.p3 = cms.Path(process.onPath+process.f1, cms.Task(process.getter))

process.p2 = cms.Path(process.onPath+process.f2)

#from FWCore.Modules.modules import EventContentAnalyzer import *
#process.dump = EventContentAnalyzer()
#process.p = cms.Path(process.dump)

process.out = AsciiOutputModule()
process.e = cms.EndPath(process.out, cms.Task(process.getter))

process.maxEvents.input = 1

#from FWCore.Services.modules import Tracer
#process.add_(Tracer())
