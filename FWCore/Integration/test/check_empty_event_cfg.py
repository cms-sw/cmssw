import FWCore.ParameterSet.Config as cms
from FWCore.Modules.modules import EventContentAnalyzer, EmptySource

process = cms.Process("EMPTY")

process.test = EventContentAnalyzer(listPathStatus = True)

process.e = cms.EndPath(process.test)

#process.out = cms.OutputModule("AsciiOutputModule")
#process.e2 = cms.EndPath(process.out)

process.source = EmptySource()

process.maxEvents.input = 1
