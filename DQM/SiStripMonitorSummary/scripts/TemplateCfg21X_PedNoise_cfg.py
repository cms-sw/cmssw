import FWCore.ParameterSet.Config as cms    

process = cms.Process("CONDOBJMON")
#-------------------------------------------------
# CALIBRATION
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.Tags21X_cff")


#-------------------------------------------------
# DQM
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondDataOffline_cfi")
process.CondDataMonitoring.OutputMEsInRootFile        = cms.bool(False)
process.CondDataMonitoring.MonitorSiStripPedestal      =cms.bool(True)
process.CondDataMonitoring.MonitorSiStripNoise         =cms.bool(True)
process.CondDataMonitoring.MonitorSiStripQuality       =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripCabling       =cms.bool(True)
process.CondDataMonitoring.MonitorSiStripLowThreshold  =cms.bool(True)
process.CondDataMonitoring.MonitorSiStripHighThreshold =cms.bool(True)
process.CondDataMonitoring.MonitorSiStripApvGain       =cms.bool(False)
process.CondDataMonitoring.MonitorSiStripLorentzAngle  =cms.bool(False)

process.load("DQM.SiStripCommon.TkHistoMap_cfi")
process.load("DQM.SiStripMonitorClient.SiStripDQMOffline_cff")

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
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(1)
)


import CalibTracker.Configuration.Common.PoolDBESSource_cfi
siStripCond = CalibTracker.Configuration.Common.PoolDBESSource_cfi.poolDBESSource.clone()

sistripconn = cms.ESProducer("SiStripConnectivity")

siStripCond.toGet = cms.VPSet(
    cms.PSet(
    record = cms.string('SiStripFedCablingRcd'),
    tag = cms.string('insert_FedCablingTag')
    ), 
    cms.PSet( 
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('insert_NoiseTag')
    ), 
    cms.PSet(
        record = cms.string('SiStripPedestalsRcd'),
        tag = cms.string('insert_PedestalTag')
    ),
    cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripGain_Ideal_21X')
    ),
    cms.PSet(
        record = cms.string('SiStripLorentzAngleRcd'),
        tag = cms.string('SiStripLorentzAngle_Ideal_21X')
    ),     
    cms.PSet(
        record = cms.string('SiStripThresholdRcd'),
        tag = cms.string('insert_ThresholdTag')
    ))


    
siStripCond.connect = 'frontier://cmsfrontier.cern.ch:8000/FrontierProd/insertAccount'


    #-------------------------------------------------
    ## Scheduling
    #-------------------------------------------------

####process.p = cms.Path(process.CondDataMonitoring*process.qTester)
process.p = cms.Path(process.CondDataMonitoring*process.SiStripOfflineDQMClient*process.qTester*process.dqmSaver)





  

