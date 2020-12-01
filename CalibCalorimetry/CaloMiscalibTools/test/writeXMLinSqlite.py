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
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string('EcalIntercalibConstants_ideal')
    ))
)

process.CaloMiscalibTools = cms.ESSource("CaloMiscalibTools",
    fileNameBarrel = cms.untracked.string('EcalIntercalibConstants_EB_ideal.xml'),
    fileNameEndcap = cms.untracked.string('EcalIntercalibConstants_EE_ideal.xml')
)

process.prefer("CaloMiscalibTools")
process.WriteInDB = cms.EDFilter("WriteEcalMiscalibConstants",
    NewTagRequest = cms.string('EcalIntercalibConstantsRcd')
)

process.p = cms.Path(process.WriteInDB)


