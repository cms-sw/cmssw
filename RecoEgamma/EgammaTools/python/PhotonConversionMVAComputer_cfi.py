from CondCore.DBCommon.CondDBSetup_cfi import *

PhotonConversionMVAComputerRcd = cms.ESSource("PoolDBESSource",
        CondDBSetup,
        toGet = cms.VPSet(cms.PSet(
            record = cms.string('PhotonConversionMVAComputerRcd'),
            tag = cms.string('some_pooldb_tag')
            )),
        connect = cms.string('sqlite_file:localconditions.db'),
        )


