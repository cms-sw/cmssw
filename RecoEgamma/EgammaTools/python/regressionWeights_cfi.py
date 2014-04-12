from CondCore.DBCommon.CondDBSetup_cfi import *

GBRWrapperRcd  =  cms.ESSource("PoolDBESSource",
    CondDBSetup,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EBCorrection'),
        label = cms.untracked.string('wgbrph_EBCorrection')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EBUncertainty'),
        label = cms.untracked.string('wgbrph_EBUncertainty')
      ),    
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EECorrection'),
        label = cms.untracked.string('wgbrph_EECorrection')
      ),
      cms.PSet(
        record = cms.string('GBRWrapperRcd'),
        tag = cms.string('wgbrph_EEUncertainty'),
        label = cms.untracked.string('wgbrph_EEUncertainty')
      )),
     connect = cms.string('sqlite_file:GBRWrapper.db')
)

