import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetriever_cfi")
process.EcalTrivialConditionRetriever.producedEcalLaserAlphas =  cms.untracked.bool(True)

process.EcalTrivialConditionRetriever.getLaserAlphaFromTypeEB =  cms.untracked.bool(True)
process.EcalTrivialConditionRetriever.getLaserAlphaFromTypeEE =  cms.untracked.bool(True)
process.EcalTrivialConditionRetriever.laserAlphaMeanEBR = cms.untracked.double(1.52)
process.EcalTrivialConditionRetriever.laserAlphaMeanEBC = cms.untracked.double(1.00)
process.EcalTrivialConditionRetriever.laserAlphaMeanEER = cms.untracked.double(1.16)
process.EcalTrivialConditionRetriever.laserAlphaMeanEEC = cms.untracked.double(1.00)
# uses CalibCalorimetry/EcalTrivialCondModules/data/EBLaserAlpha.txt for Russian/Chinese Xtals distinction
# uses CalibCalorimetry/EcalTrivialCondModules/data/EELaserAlpha.txt for Russian/Chinese Xtals distinction

process.EcalTrivialConditionRetriever.getLaserAlphaFromFileEB =  cms.untracked.bool(False)
process.EcalTrivialConditionRetriever.getLaserAlphaFromFileEE =  cms.untracked.bool(False)
#process.EcalTrivialConditionRetriever.EBLaserAlphaFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/EBLaserAlpha_fromFile.txt')
#process.EcalTrivialConditionRetriever.EELaserAlphaFile = cms.untracked.string('CalibCalorimetry/EcalTrivialCondModules/data/EELaserAlpha_fromFile.txt')
# ieta, iphi, alpha for EB
# iz, ix, iy, alpha for EE

process.load("CondCore.CondDB.CondDB_cfi")
# process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_ECAL'
process.CondDB.connect = 'sqlite_file:EcalLaserAlphas_byTYPE.db'
# process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'


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
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalLaserAlphasRcd'),
        tag = cms.string('EcalLaserAlphas_byTYPE')
        ))
)

process.dbCopy = cms.EDAnalyzer("EcalDBCopy",
    timetype = cms.string('runnumber'),
    toCopy = cms.VPSet(cms.PSet(
        record = cms.string('EcalLaserAlphasRcd'),
        container = cms.string('EcalLaserAlphas')
        ))
)



process.p = cms.Path(process.dbCopy)

