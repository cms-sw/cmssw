import FWCore.ParameterSet.Config as cms    

process = cms.Process("CONDOBJMON")
#-------------------------------------------------

#-------------------------------------------------
# DQM
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondDataOffline_cfi")
process.CondDataMonitoring.OutputMEsInRootFile        = cms.bool(True)
process.CondDataMonitoring.MonitorSiStripPedestal      =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripNoise         =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripQuality       =cms.bool(True)
process.CondDataMonitoring.MonitorSiStripCabling       =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripLowThreshold  =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripHighThreshold =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripApvGain       =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripLorentzAngle  =cms.bool(False)

## SI STRIP MONITOR TRACK:
process.load("DQM.SiStripCommon.TkHistoMap_cfi");

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(insert_runnumber),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(insert_runnumber),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiStripMonitorCondData'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('Error')
    ),
    destinations = cms.untracked.vstring('error.log', 
        'cout')
)

process.qTester = cms.EDFilter("QualityTester",
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorSummary/data/insert_QtestsFileName'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)

process.DQMStore = cms.Service("DQMStore",
    verbose = cms.untracked.int32(1)
)

from CalibTracker.Configuration.Common.PoolDBESSource_cfi import *
siStripCond = poolDBESSource.clone(
    toGet = (
        poolDBESSource.toGet[0].clone(
            record = 'SiStripFedCablingRcd',
            tag = 'insert_FedCablingTag'
        ), 
        poolDBESSource.toGet[0].clone( 
            record = 'SiStripNoisesRcd',
            tag = 'insert_NoiseTag'
        ), 
        poolDBESSource.toGet[0].clone(
            record = 'SiStripPedestalsRcd',
            tag = 'insert_PedestalTag'
        ),
        poolDBESSource.toGet[0].clone(
            record = 'SiStripApvGainRcd',
            tag = 'SiStripGain_Ideal_21X'
        ),
        poolDBESSource.toGet[0].clone(
            record = 'SiStripLorentzAngleRcd',
            tag = 'SiStripLorentzAngle_Ideal_21X'
        ),     
        poolDBESSource.toGet[0].clone(
            record = 'SiStripThresholdRcd',
            tag = 'insert_ThresholdTag'
        )
    ),
    connect = 'frontier://cmsfrontier.cern.ch:8000/FrontierProd/insertAccount'
)

sistripconn = cms.ESProducer("SiStripConnectivity")

process.a = cms.ESSource("PoolDBESSource",
    appendToDataLabel = cms.string('test'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
##        record = cms.string('SiStripDetCablingRcd'),
        tag = cms.string('insert_DB_Tag')
    )
    ),
    DBParameters = cms.PSet(
    authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    connect = cms.string('frontier://cmsfrontier.cern.ch:8000/FrontierProd/insertAccount'),
)

process.MySSQ = cms.ESProducer("SiStripQualityESProducer",
    appendToDataLabel = cms.string(''),
    ReduceGranularity = cms.bool(True),
    ThresholdForReducedGranularity = cms.double(0.3),
    ListOfRecordToMerge = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadFiberRcd'),
##        record = cms.string('SiStripDetCablingRcd'),
##        record = cms.string('SiStripBadChannelRcd'),        
        tag = cms.string('test')
    ))
)

#process.MySSQPrefer = cms.ESPrefer("PoolDBESSource","a")


    #-------------------------------------------------
    ## Scheduling
    #-------------------------------------------------

process.p = cms.Path(process.CondDataMonitoring*process.qTester)
####process.p = cms.Path(process.CondDataMonitoring)





  

