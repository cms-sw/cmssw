import FWCore.ParameterSet.Config as cms
from CondCore.DBCommon.CondDBCommon_cfi import *
CondDBCommon.connect = 'sqlite_file:JEC_Summer09_7TeV_ReReco332.db'

PoolDBESSource = cms.ESSource("PoolDBESSource",
  CondDBCommon,
  toGet = cms.VPSet( 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_IC5JPT'), 
         label  = cms.untracked.string('L2Relative_IC5JPT') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_IC5Calo'), 
         label  = cms.untracked.string('L2Relative_IC5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_IC5PF'), 
         label  = cms.untracked.string('L2Relative_IC5PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_AK5JPT'), 
         label  = cms.untracked.string('L2Relative_AK5JPT') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_AK5Calo'), 
         label  = cms.untracked.string('L2Relative_AK5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_AK5PF'), 
         label  = cms.untracked.string('L2Relative_AK5PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_AK7Calo'), 
         label  = cms.untracked.string('L2Relative_AK7Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_AK7PF'), 
         label  = cms.untracked.string('L2Relative_AK7PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_KT4Calo'), 
         label  = cms.untracked.string('L2Relative_KT4Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_KT4PF'), 
         label  = cms.untracked.string('L2Relative_KT4PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_KT6Calo'), 
         label  = cms.untracked.string('L2Relative_KT6Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L2Relative_KT6PF'), 
         label  = cms.untracked.string('L2Relative_KT6PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_IC5JPT'), 
         label  = cms.untracked.string('L3Absolute_IC5JPT') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_IC5Calo'), 
         label  = cms.untracked.string('L3Absolute_IC5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_IC5PF'), 
         label  = cms.untracked.string('L3Absolute_IC5PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_AK5JPT'), 
         label  = cms.untracked.string('L3Absolute_AK5JPT') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_AK5Calo'), 
         label  = cms.untracked.string('L3Absolute_AK5Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_AK5PF'), 
         label  = cms.untracked.string('L3Absolute_AK5PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_AK7Calo'), 
         label  = cms.untracked.string('L3Absolute_AK7Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_AK7PF'), 
         label  = cms.untracked.string('L3Absolute_AK7PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_KT4Calo'), 
         label  = cms.untracked.string('L3Absolute_KT4Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_KT4PF'), 
         label  = cms.untracked.string('L3Absolute_KT4PF') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_KT6Calo'), 
         label  = cms.untracked.string('L3Absolute_KT6Calo') 
      ), 
      cms.PSet( 
         record = cms.string('JetCorrectionsRecord'), 
         tag    = cms.string('L3Absolute_KT6PF'), 
         label  = cms.untracked.string('L3Absolute_KT6PF') 
      )
  )
)
