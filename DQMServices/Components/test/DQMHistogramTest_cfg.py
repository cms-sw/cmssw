import os
import time
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('runNumber',
                 4294967292, #default value, int limit -3
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number; default gives latest IOV")
options.register('eventsPerLumi',
                 3, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of events per lumi")
options.register('numberOfLumis',
                 3, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of lumisections per run")
options.register('numberOfRuns',
                 3, #default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "number of runs in the job")

options.parseArguments()

process = cms.Process('RECODQM')

histoRunList = ["Foo"]
histoLumiList = ["bAr"]

#load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#my analyzer
process.load('DQMServices.Components.DQMHistogramTest_cfi')
[process.dqmHistogramTest.histograms.append( x ) for x in histoLumiList]
[process.dqmHistogramTest.histograms.append( x ) for x in histoRunList]

#Database Dump
process.load("DQMServices.Components.DQMHistogramDB_cfi")
[process.dqmHistogramDB.histogramNamesEndLumi.append( os.path.join(process.dqmHistogramTest.path.value(), x) ) for x in histoLumiList]
[process.dqmHistogramDB.histogramNamesEndRun.append( os.path.join(process.dqmHistogramTest.path.value(), x) ) for x in histoRunList]

# Input source
process.source = cms.Source( "EmptySource",
                             firstRun = cms.untracked.uint32( options.runNumber ),
                             firstTime = cms.untracked.uint64( ( long( time.time() ) - 24 * 3600 ) << 32 ), #24 hours ago in nanoseconds
                             numberEventsInRun = cms.untracked.uint32( options.eventsPerLumi *  options.numberOfLumis ), # options.numberOfLumis lumi sections per run
                             numberEventsInLuminosityBlock = cms.untracked.uint32( options.eventsPerLumi )
                             )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32( options.eventsPerLumi * options.numberOfLumis * options.numberOfRuns ) ) #options.numberOfRuns runs per job

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT_step1.root"))

# Path and EndPath definitions
process.dqmoffline_step = cms.Path(process.dqmHistogramTest)
process.DQMoutput_step = cms.EndPath(process.dqmHistogramDB*process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(
    process.dqmoffline_step,
    process.DQMoutput_step
)
