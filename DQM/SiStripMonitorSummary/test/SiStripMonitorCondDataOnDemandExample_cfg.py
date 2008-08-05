import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#-------------------------------------------------
# CALIBRATION
#-------------------------------------------------
#include "DQM/SiStripMonitorSummary/data/Tags20X.cff"
process.load("CalibTracker.Configuration.Tracker_FrontierConditions_TIF_20X_cff")

#-------------------------------------------------
# DQM
#-------------------------------------------------
process.load("DQM.SiStripMonitorSummary.SiStripMonitorCondDataOnDemandExample_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('SiStripMonitorCondDataOnDemandExample'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('Error')
    ),
    destinations = cms.untracked.vstring('error.log', 
        'cout')
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/TAC/TIBTOB/edm_2007_03_07/tif.00006215.A.testStorageManager_0.0.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)
process.DQMStore = cms.Service("DQMStore",
    referenceFileName = cms.untracked.string(''),
    verbose = cms.untracked.int32(0)
)

process.p = cms.Path(process.myOnDemandExample)
process.myOnDemandExample.OutputMEsInRootFile = True
process.myOnDemandExample.MonitorSiStripPedestal = True
process.myOnDemandExample.MonitorSiStripNoise = True
process.myOnDemandExample.MonitorSiStripQuality = False
process.myOnDemandExample.MonitorSiStripApvGain = True
process.myOnDemandExample.MonitorSiStripLorentzAngle = True

