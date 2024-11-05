#! /usr/bin/env cmsRun
# Author: Marco Musich (October 2021)
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

###################################################################
# Set default phase-2 settings
###################################################################
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

process = cms.Process("TEST",_PH2_ERA)
options = VarParsing.VarParsing('analysis')
options.register('fromESSource',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Populate SiPhase2OuterTrackerLorentzAngleRcd from the ESSource")
options.parseArguments()

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')   
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2OuterTrackerLorentzAngleReader=dict()  
process.MessageLogger.SiPhase2OuterTrackerLorentzAngle=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),                                                      
  SiPhase2OuterTrackerLorentzAngleReader = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
  SiPhase2OuterTrackerLorentzAngle = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
  SiPhase2OuterTrackerFakeLorentzAngleESSource =  cms.untracked.PSet( limit = cms.untracked.int32(-1))
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
    process.load("Configuration.Geometry.GeometryExtendedRun4Default_cff")
    process.load('Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff')
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    from Configuration.AlCa.GlobalTag import GlobalTag
    process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, '')

    # process.SiPhase2OTFakeLorentzAngleESSource = cms.ESSource('SiPhase2OuterTrackerFakeLorentzAngleESSource',
    #                                                           LAValue = cms.double(0.014),
    #                                                           recordName = cms.string("LorentzAngle"))
    # process.es_prefer_fake_LA = cms.ESPrefer("SiPhase2OuterTrackerFakeLorentzAngleESSource","SiPhase2OTFakeLorentzAngleESSource")

    from CalibTracker.SiPhase2TrackerESProducers.SiPhase2OuterTrackerFakeLorentzAngleESSource_cfi import SiPhase2OTFakeLorentzAngleESSource
    process.mySiPhase2OTFakeLorentzAngleESSource =  SiPhase2OTFakeLorentzAngleESSource.clone(LAValue = cms.double(0.14))
    process.es_prefer_fake_LA = cms.ESPrefer("SiPhase2OuterTrackerFakeLorentzAngleESSource","mySiPhase2OTFakeLorentzAngleESSource")
else:
    tag = 'SiPhase2OuterTrackerLorentzAngle_T33'
    suffix = 'v0'
    inFile = tag+'_'+suffix+'.db'
    inDB = 'sqlite_file:'+inFile

    process.load("CondCore.CondDB.CondDB_cfi")
    # input database (in this case the local sqlite file)
    process.CondDB.connect = inDB

    process.PoolDBESSource = cms.ESSource("PoolDBESSource",
                                          process.CondDB,
                                          DumpStat=cms.untracked.bool(True),
                                          toGet = cms.VPSet(cms.PSet(record = cms.string("SiPhase2OuterTrackerLorentzAngleRcd"),
                                                                     tag = cms.string(tag))
                                                            )
                                          )

###################################################################
# check the ES data getter
###################################################################
process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiPhase2OuterTrackerLorentzAngleRcd'),
        data = cms.vstring('SiPhase2OuterTrackerLorentzAngle')
    )),
    verbose = cms.untracked.bool(True)
)

###################################################################
# Payload reader
###################################################################
import CondTools.SiPhase2Tracker.siPhase2OuterTrackerLorentzAngleReader_cfi as _mod
process.LAPayloadReader = _mod.siPhase2OuterTrackerLorentzAngleReader.clone(printDebug = 10,
                                                                            label = "")

###################################################################
# Path
###################################################################
process.p = cms.Path(process.LAPayloadReader)

