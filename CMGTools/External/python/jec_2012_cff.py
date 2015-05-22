from CondCore.DBCommon.CondDBSetup_cfi import *

## Preview JEC 2012
def use2012JecPreview(process,data=False):
    process.load("CondCore.DBCommon.CondDBCommon_cfi")
    if data:
        jec_db="Summer12_V1_DATA.db"
        ret='Summer12_V1_DATA'
        tag='JetCorrectorParametersCollection_Summer12_V1_DATA_AK5PF'
    else:
        jec_db="Summer12_V1_MC.db"
        ret='Summer12_V1_MC'
        tag='JetCorrectorParametersCollection_Summer12_V1_MC_AK5PF'
    process.jec = cms.ESSource("PoolDBESSource",
                               DBParameters = cms.PSet(messageLevel = cms.untracked.int32(0)
                                                       ),
                               timetype = cms.string('runnumber'),
                               toGet = cms.VPSet(
        cms.PSet(
        record = cms.string('JetCorrectionsRecord'),
        ## tag    = cms.string('JetCorrectorParametersCollection_Jec12_V7_AK5PF'),
        tag    = cms.string(tag),
        label  = cms.untracked.string('AK5PF')
        ),
        ),
                               
                               connect = cms.string('sqlite_fip:CMGTools/External/data/%s' % jec_db)
                               )
    process.es_prefer_jec = cms.ESPrefer('PoolDBESSource','jec')
    return ret
