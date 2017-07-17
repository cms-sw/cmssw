import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource",)
        

process.gbrwrappermaker = cms.EDAnalyzer('GBRWrapperMaker')
        
#Database output service
process.load("CondCore.DBCommon.CondDBCommon_cfi")
# output database (in this case local sqlite file)
process.CondDBCommon.connect = 'sqlite_file:mustacheSC_online_regression_18012015.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(
        cms.PSet(
            record = cms.string('mustacheSC_online_EBCorrection'),
            tag    = cms.string('mustacheSC_online_EBCorrection')
            ),
        cms.PSet(
            record = cms.string('mustacheSC_online_EBUncertainty'),
            tag    = cms.string('mustacheSC_online_EBUncertainty')
            ),
        cms.PSet(
            record = cms.string('mustacheSC_online_EECorrection'),
            tag    = cms.string('mustacheSC_online_EECorrection')
            ),
        cms.PSet(
            record = cms.string('mustacheSC_online_EEUncertainty'),
            tag    = cms.string('mustacheSC_online_EEUncertainty')
            ),
        
        )
                                          )
            
                
                    
process.p = cms.Path(process.gbrwrappermaker)
