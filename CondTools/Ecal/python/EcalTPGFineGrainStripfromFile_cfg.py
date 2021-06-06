import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#process.load("CondCore.DBCommon.CondDB_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
#
# Choose the output database
#
process.CondDB.connect = 'sqlite_file:EcalTPGFineGrainStrip.db'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
  firstValue = cms.uint64(1),
  lastValue = cms.uint64(1),
  timetype = cms.string('runnumber'),
  interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB,
  logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalTPGFineGrainStripEERcd'),
      tag = cms.string('EcalTPGFineGrainStripEE_test_10GeV')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalTPGFineGrainStripfromFile",
  record = cms.string('EcalTPGFineGrainStripEERcd'),
  Source = cms.PSet(
    FileName = cms.string('/afs/cern.ch/cms/CAF/CMSCOMM/COMM_ECAL/azabi/TPG_beamv6_trans_spikekill_FGEE_10GEV_LUT.txt')
  )
)

process.p = cms.Path(process.Test1)
