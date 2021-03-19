# The following comments couldn't be translated into the new config version:

# Database output service

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#PopCon config
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:DBChipSpeedCorrection_data_test.db")
#process.CondDBCommon.connect = cms.string("sqlite_file:CSC_chip_and_chamber_time_corr_MC.db")
#process.CondDBCommon.connect = cms.string("oracle://cms_orcoff_prep/CMS_COND_CSC")
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        enable = cms.untracked.bool(True)
    )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    #change the firstRun if you want a different IOV
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:gainslog.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CSCDBChipSpeedCorrectionRcd'),
        tag = cms.string('CSCDBChipSpeedCorrection') 
    ))
)

process.WriteChipSpeedCorrectionWithPopCon = cms.EDAnalyzer("CSCDBChipSpeedCorrectionPopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('CSCDBChipSpeedCorrectionRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
         # File supplied by S. Durkin 11 May 2010
         dataCorrFileName= cms.untracked.string("/afs/cern.ch/user/d/deisher/public/TimingCorrections2009/CathodeTimingCorrection_11May2010.txt"),
         isForMC = cms.untracked.bool(False)
    )
)

process.p = cms.Path(process.WriteChipSpeedCorrectionWithPopCon)
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_CSC'



