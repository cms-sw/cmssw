import FWCore.ParameterSet.Config as cms
from CondCore.DBCommon.CondDBCommon_cfi import *
#CondDBCommon.connect = 'frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS'
CondDBCommon.connect = cms.string('sqlite_file:JEC_Spring10.db')


PoolDBESSource = cms.ESSource("PoolDBESSource",
  CondDBCommon,
  toGet = cms.VPSet( 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JEC_Spring10_AK5Calo'), 
         label  = cms.untracked.string('JEC_Spring10_AK5Calo') 
      )
  )
)
PoolDBESSource = cms.ESSource("PoolDBESSource",
  CondDBCommon,
  toGet = cms.VPSet( 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JEC_Spring10_AK5PF'), 
         label  = cms.untracked.string('JEC_Spring10_AK5PF') 
      )
  )
)
PoolDBESSource = cms.ESSource("PoolDBESSource",
  CondDBCommon,
  toGet = cms.VPSet( 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('JEC_Summer10_AK5JPT'), 
         label  = cms.untracked.string('JEC_Summer10_AK5JPT') 
      )
  )
)
