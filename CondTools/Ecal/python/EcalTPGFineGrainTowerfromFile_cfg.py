import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
#process.load("CondCore.DBCommon.CondDB_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'
#
# Choose the output database
#
process.CondDB.connect = 'sqlite_file:EcalTPGFineGrainTower.db'

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

#process.PoolDBESSource = cms.ESSource("PoolDBESSource",
#  process.CondDB,
#  timetype = cms.untracked.string('runnumber'),
#  toGet = cms.VPSet(
#    cms.PSet(
#      record = cms.string('EcalTPGFineGrainTowerEERcd'),
#      tag = cms.string('EcalTPGFineGrainTower_test')
#    )
#  )
# )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
  process.CondDB,
  logconnect = cms.untracked.string('sqlite_file:DBLog.db'),
  timetype = cms.untracked.string('runnumber'),
  toPut = cms.VPSet(
    cms.PSet(
      record = cms.string('EcalTPGFineGrainTowerEERcd'),
      tag = cms.string('EcalTPGFineGrainTower_test')
    )
  )
)

process.Test1 = cms.EDAnalyzer("ExTestEcalTPGFineGrainTowerfromFile",
  record = cms.string('EcalTPGFineGrainTowerEERcd'),
  Source = cms.PSet(
#    debug = cms.bool(True),
    FileName = cms.string('/afs/cern.ch/cms/CAF/CMSCOMM/COMM_ECAL/azabi/TPG_beamv6_trans_spikekill_FGEE_2GEV_LUT.txt')
  )
)

process.p = cms.Path(process.Test1)
