from CondCore.DBCommon.CondDBSetup_cfi import *
 
## mvaPFMEtGBRForestsFromDB = cms.ESSource("PoolDBESSource",
##     CondDBSetup,
##     DumpStat = cms.untracked.bool(True),
##     toGet = cms.VPSet(
##         cms.PSet(
##             record = cms.string('GBRWrapperRcd'),
##             tag = cms.string('mvaPFMET_53_Dec2012_U'),
##             label = cms.untracked.string('mvaPFMET_53_Dec2012_U')
##         ),
##         cms.PSet(
##             record = cms.string('GBRWrapperRcd'),
##             tag = cms.string('mvaPFMET_53_Dec2012_DPhi'),
##             label = cms.untracked.string('mvaPFMET_53_Dec2012_DPhi')
##         ),
##         cms.PSet(
##             record = cms.string('GBRWrapperRcd'),
##             tag = cms.string('mvaPFMET_53_Dec2012_CovU1'),
##             label = cms.untracked.string('mvaPFMET_53_Dec2012_CovU1')
##         ),
##         cms.PSet(
##             record = cms.string('GBRWrapperRcd'),
##             tag = cms.string('mvaPFMET_53_Dec2012_CovU2'),
##             label = cms.untracked.string('mvaPFMET_53_Dec2012_CovU2')
##         )
##     ),
##     connect = cms.string('sqlite_fip:RecoMET/METPUSubtraction/data/mvaPFMEt_53_Dec2012.db')
## )
