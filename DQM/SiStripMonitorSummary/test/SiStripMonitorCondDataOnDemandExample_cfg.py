import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#-------------------------------------------------
# CALIBRATION
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.Tags21X_cff")

#-------------------------------------------------
# DQM
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondDataOnDemandExample_cfi")

## SI STRIP MONITOR TRACK:
process.load("DQM.SiStripCommon.TkHistoMap_cfi");

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiStripMonitorCondDataOnDemandExample'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('Error')
    ),
    destinations = cms.untracked.vstring('error.log', 
        'cout')
)


process.source = cms.Source("EmptySource",
    lastRun = cms.untracked.uint32(70409),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(70409),
    interval = cms.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester = DQMQualityTester(
    qtList = cms.untracked.FileInPath('DQM/SiStripMonitorSummary/data/CondDBQtests.xml'),
    QualityTestPrescaler = cms.untracked.int32(1),
    getQualityTestsFromFile = cms.untracked.bool(True)
)


process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(1)
)

process.p = cms.Path(process.myOnDemandExample*process.qTester)
process.myOnDemandExample.OutputMEsInRootFile = True

process.myOnDemandExample.MonitorSiStripPedestal      = False
process.myOnDemandExample.MonitorSiStripNoise         = False
process.myOnDemandExample.MonitorSiStripApvGain       = False ## to be tested on REAL data

process.myOnDemandExample.MonitorSiStripQuality      = True
process.myOnDemandExample.MonitorSiStripLorentzAngle = False
process.myOnDemandExample.MonitorSiStripBackPlaneCorrection = False
process.myOnDemandExample.MonitorSiStripCabling      = False



process.myOnDemandExample.SiStripNoisesDQM_PSet.GainRenormalisation   = False
process.myOnDemandExample.SiStripNoisesDQM_PSet.SimGainRenormalisation   = False
process.myOnDemandExample.SiStripNoisesDQM_PSet.CondObj_fillId        = 'onlyProfile'
process.myOnDemandExample.SiStripPedestalsDQM_PSet.CondObj_fillId        = 'onlyProfile'
