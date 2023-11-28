#! /usr/bin/env cmsRun
# Author: Marco Musich (Feb 2022)
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("TEST")
options = VarParsing.VarParsing('analysis')
options.register('fromESSource',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Populate SiPhase2OuterTrackerBadStripRcd from the ESSource")
options.parseArguments()

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2BadStripChannelReader=dict()  
process.MessageLogger.SiStripBadStrip=dict()
process.MessageLogger.SiPhase2BadStripConfigurableFakeESSource=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
    SiPhase2BadStripChannelReader = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiStripBadStrip = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiPhase2BadStripConfigurableFakeESSource = cms.untracked.PSet( limit = cms.untracked.int32(-1))
)

###################################################################
# A data source must always be defined.
# We don't need it, so here's a dummy one.
###################################################################
process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

###################################################################
# Input data
###################################################################
if(options.fromESSource): 
    process.load("Configuration.Geometry.GeometryExtended2026D88_cff")
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')
    #process.load("SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff") # already included
    #process.SiPhase2OTFakeBadStripsESSource.printDebug = cms.untracked.bool(True)    # this makes it verbose 
    process.SiPhase2OTFakeBadStripsESSource.badComponentsFraction = cms.double(0.05)   # bad components fraction is 5%
else:
    tag = 'SiStripBadStripPhase2_T21'
    suffix = 'v0'
    inFile = tag+'_'+suffix+'.db'
    inDB = 'sqlite_file:'+inFile
    
    process.load("CondCore.CondDB.CondDB_cfi")
    # input database (in this case the local sqlite file)
    process.CondDB.connect = inDB
    
    process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                          process.CondDB,
                                          DumpStat=cms.untracked.bool(True),
                                          toGet = cms.VPSet(cms.PSet(#record = cms.string("SiStripBadStripRcd"),
                                                                     record = cms.string("SiPhase2OuterTrackerBadStripRcd"),
                                                                     tag = cms.string(tag))))
    
###################################################################
# check the ES data getter
###################################################################
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        #record = cms.string('SiStripBadStripRcd'),
        record = cms.string('SiPhase2OuterTrackerBadStripRcd'),
        data = cms.vstring('SiStripBadStrip')
    )),
    verbose = cms.untracked.bool(True)
)

###################################################################
# Payload reader
###################################################################
import CondTools.SiPhase2Tracker.siPhase2BadStripChannelReader_cfi as _mod
process.BadStripPayloadReader = _mod.siPhase2BadStripChannelReader.clone(printDebug = 1,
                                                                         printVerbose = False,
                                                                         label = "")

###################################################################
# Path
###################################################################
process.p = cms.Path(process.get+process.BadStripPayloadReader)
