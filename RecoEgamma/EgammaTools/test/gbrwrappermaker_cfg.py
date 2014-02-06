import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.source = cms.Source("PoolSource",
    ## replace 'myfile.root' with the source file you want to use
    #fileNames = cms.untracked.vstring(
        #'file:myfile.root'
    #)
#)

process.source = cms.Source("EmptySource",
)
        

process.gbrwrappermaker = cms.EDAnalyzer('GBRWrapperMaker'
)


    
        
#Database output service

process.load("CondCore.DBCommon.CondDBCommon_cfi")
# output database (in this case local sqlite file)
process.CondDBCommon.connect = 'sqlite_file:gedelectron_p4combination_14122013.db'


process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(
      cms.PSet(
        record = cms.string('gedelectron_p4combination'),
        tag = cms.string('gedelectron_p4combination')
      )
      
  )
)
            
                
                    
process.p = cms.Path(process.gbrwrappermaker)
