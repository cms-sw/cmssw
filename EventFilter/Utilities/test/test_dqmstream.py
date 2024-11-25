import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
from FWCore.ParameterSet.Types import PSet

process = cms.Process("DQMTEST")

options = VarParsing.VarParsing('analysis')

options.register('runNumber',
                 100101,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run number.")

options.register('runInputDir',
                 '/tmp',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Directory where the DQM files will appear.")
options.register('eventsPerLS',
                 35,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Max LS to generate (0 to disable limit)")                 
options.register ('maxLS',
                  2,
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "Max LS to generate (0 to disable limit)")

options.parseArguments()


process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring('cout'),
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('WARNING'))
                                    )
process.source = cms.Source("DQMStreamerReader",
        runNumber = cms.untracked.uint32(options.runNumber),
        runInputDir = cms.untracked.string(options.runInputDir),
        streamLabel = cms.untracked.string('streamDQM'),
        scanOnce = cms.untracked.bool(True),
        minEventsPerLumi = cms.untracked.int32(1),
        delayMillis = cms.untracked.uint32(500),
        nextLumiTimeoutMillis = cms.untracked.int32(0),
        skipFirstLumis = cms.untracked.bool(False),
        deleteDatFiles = cms.untracked.bool(False),
        endOfRunKills  = cms.untracked.bool(False),
        inputFileTransitionsEachEvent = cms.untracked.bool(False),
        SelectEvents = cms.untracked.vstring("HLT*Mu*","HLT_*Physics*")
)

#make a list of all the EventIDs that were seen by the previous job,
# given the filter is semi-random we do not know which of these will
# be the actual first event written
rn = options.runNumber
transitions = [cms.EventID(rn,0,0)]
evid = 1
for lumi in range(1, options.maxLS+1):
    transitions.append(cms.EventID(rn,lumi,0))
    for ev in range(0, options.eventsPerLS):
        transitions.append(cms.EventID(rn,lumi,evid))
        evid += 1
    transitions.append(cms.EventID(rn,lumi,0)) #end lumi
transitions.append(cms.EventID(rn,0,0)) #end run


#only see 1 event as process.source.minEventsPerLumi == 1
process.test = cms.EDAnalyzer("RunLumiEventChecker",
                              eventSequence = cms.untracked.VEventID(*transitions),
                              unorderedEvents = cms.untracked.bool(True),
                              minNumberOfEvents = cms.untracked.uint32(1+2+2),
                              maxNumberOfEvents = cms.untracked.uint32(1+2+2)
)
if options.eventsPerLS == 0:
    process.test.eventSequence = []
    process.test.minNumberOfEvents = 0
    process.test.maxNumberOfEvents = 0
    
process.p = cms.Path(process.test)
                              
