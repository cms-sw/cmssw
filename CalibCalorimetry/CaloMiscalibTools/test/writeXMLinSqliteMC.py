import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:DB.db'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

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
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsMCRcd'),
        tag = cms.string('EcalIntercalibConstantsMC_inv_startup')
    ))
)

process.CaloMiscalibToolsMC = cms.ESSource("CaloMiscalibToolsMC",
    fileNameBarrel = cms.untracked.string('inv_EcalIntercalibConstants_EB_startup.xml'),
    fileNameEndcap = cms.untracked.string('inv_EcalIntercalibConstants_EE_startup.xml')
)

process.prefer("CaloMiscalibToolsMC")
process.WriteInDB = cms.EDFilter("WriteEcalMiscalibConstantsMC",
    NewTagRequest = cms.string('EcalIntercalibConstantsMCRcd')
)

process.p = cms.Path(process.WriteInDB)


