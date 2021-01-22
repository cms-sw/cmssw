import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:ESGain_LG.db'

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
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(
    # gain
    record = cms.string('ESGainRcd'),
    tag = cms.string('ESGain_LG')
    
    )))

process.ecalModule = cms.EDAnalyzer("StoreESCondition",
                                    logfile = cms.string('./logfile.log'),
                                    gain = cms.uint32(1),
                                    toPut = cms.VPSet(cms.PSet(

    # gain
    conditionType = cms.untracked.string('ESGain'),
    since = cms.untracked.uint32(1),
    inputFile = cms.untracked.string('CondTools/Ecal/test/preshower/ESGain_LG.txt')

    )))
    
process.p = cms.Path(process.ecalModule)
    
    

