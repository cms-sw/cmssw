#
# Produce DQM plots that compare calibration constants
# from two different databases or sqlite files
#
# author: Stefano Argiro
# revision $Id: testEcalCalibrationsDQM.py,v 1.1 2009/03/27 16:08:03 argiro Exp $
#



import FWCore.ParameterSet.Config as cms

process = cms.Process("ecalCalibDQM")
process.load("CondCore.DBCommon.CondDBSetup_cfi")


process.load("DQMServices.Core.DQM_cfg")


poolDBESSource.connect = "frontier://FrontierDev/CMS_COND_ALIGNMENT"
poolDBESSource.toGet = cms.VPSet(cms.PSet(
        record = cms.string('GlobalPositionRcd'),
        tag = cms.string('IdealGeometry')
    )) 
process.glbPositionSource = poolDBESSource


process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.calibRef = cms.ESSource("PoolDBESSource",
    process.CondDBSetup,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string(''),
        connect = cms.untracked.string(''),
        label = cms.untracked.string('db1')
    ), 
        cms.PSet(
            record = cms.string('EcalIntercalibConstantsRcd'),
            tag = cms.string(''),
            connect = cms.untracked.string(''),
            label = cms.untracked.string('db2')
        )),
    connect = cms.string('')
)


process.monitorEcalCalib = cms.EDFilter("DQMEcalCalibConstants",
                                  FolderName=cms.untracked.string(""),
                                  SaveToFile=cms.untracked.bool(True),
                                  FileName=cms.untracked.string("c.root"),
                                  DBlabel=cms.untracked.string("db1"),
                                  RefDBlabel=cms.untracked.string("db2"),      
                                  )

process.p = cms.Path(process.monitorEcalCalib)
process.DQM.collectorHost = ''


