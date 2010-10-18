import FWCore.ParameterSet.Config as cms
from CondCore.DBCommon.CondDBCommon_cfi import *
CondDBCommon.connect = 'frontier://FrontierPrep/CMS_COND_PHYSICSTOOLS'
#CondDBCommon.connect = cms.string('sqlite_file:JEC_Spring10.db')


PoolDBESSource = cms.ESSource("PoolDBESSource",
  CondDBCommon,
  toGet = cms.VPSet( 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('AK5Calo'), 
         label  = cms.untracked.string('AK5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('AK5PF'), 
         label  = cms.untracked.string('AK5PF') 
      ),
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('AK5JPT'), 
         label  = cms.untracked.string('AK5JPT') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('AK5TRK'), 
         label  = cms.untracked.string('AK5TRK') 
      )
  )
)
