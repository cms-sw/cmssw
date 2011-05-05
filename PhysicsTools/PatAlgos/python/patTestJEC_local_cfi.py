import FWCore.ParameterSet.Config as cms

from CondCore.DBCommon.CondDBCommon_cfi import *
jec = cms.ESSource("PoolDBESSource",
      DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
        ),
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
      cms.PSet(
            record = cms.string('JetCorrectionsRecord'),
            tag    = cms.string('JetCorrectorParametersCollection_Jec43x_AK5PF'),
            label  = cms.untracked.string('AK5PF')
            )
      ),
      ## here you add as many jet types as you need (AK5Calo, AK5JPT, AK7PF, AK7Calo, KT4PF, KT4Calo, KT6PF, KT6Calo)
      connect = cms.string('sqlite_file:Jec43x.db')
)

es_prefer_jec = cms.ESPrefer('PoolDBESSource','jec')
