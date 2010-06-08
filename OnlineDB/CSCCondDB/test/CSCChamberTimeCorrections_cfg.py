# The following comments couldn't be translated into the new config version:

# eg to write payload to the oracle database 
#   replace CondDBCommon.connect = "oracle://cms_orcoff_prep/CMS_COND_CSC"
# Database output service

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#PopCon config
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:CSCChamberTimeCorrections_data_new.db")
#process.CondDBCommon.connect = cms.string("sqlite_file:/afs/cern.ch/user/d/deisher/CMSSW_3_7_0_pre3/src/CalibMuon/CSCCalibration/test/test_chipCorr_and_chamberCorr_before_133826.db")
#process.CondDBCommon.connect = cms.string("oracle://cms_orcoff_prep/CMS_COND_CSC")
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    destinations = cms.untracked.vstring('cout')
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
    logconnect = cms.untracked.string('sqlite_file:cables.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CSCChamberTimeCorrectionsRcd'),
        tag = cms.string('CSCChamberTimeCorrections')
    ))
)

process.WriteInDB = cms.EDAnalyzer("CSCChamberTimeCorrectionsPopConAnalyzer",
    SinceAppendMode = cms.bool(True),
    record = cms.string('CSCChamberTimeCorrectionsRcd'),
    loggingOn = cms.untracked.bool(True),
    Source = cms.PSet(
         isForMC = cms.untracked.bool(True)
    )
)

process.p = cms.Path(process.WriteInDB)


