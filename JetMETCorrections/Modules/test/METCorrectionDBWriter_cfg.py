import FWCore.ParameterSet.Config as cms 
process = cms.Process('metdb')

#process.MessageLogger = cms.Service("MessageLogger",
#    destinations   = cms.untracked.vstring(
#      'myDebugOutputFile'
#      ),
#    myDebugOutputFile       = cms.untracked.PSet(
#      threshold = cms.untracked.string('DEBUG'),
#      default = cms.untracked.PSet(
#	limit = cms.untracked.int32(-1)
#	),
#      ),
#    debugModules = cms.untracked.vstring(
#      #'*')
#      'dbWriterXYshift')
#    )


process.load('CondCore.CondDB.CondDB_cfi') 
#process.load('CondCore.DBCommon.CondDBCommon_cfi') 
#process.CondDB.connect = 'sqlite_file:Summer16_V0_DATA_MEtXY.db' 
process.CondDB.connect = 'sqlite_file:Summer16_V0_MC_MEtXY.db' 
#process.CondDBCommon.connect = 'sqlite_file:MET15V0.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDB, 
   #process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         #record = cms.string('MEtXYcorrectRecord'), not working 
         record = cms.string('PfType1Met'), 
         #tag    = cms.string('MEtXYcorrectParametersCollection_Summer16_V0_DATA_PfType1Met'), 
         tag    = cms.string('MEtXYcorrectParametersCollection_Summer16_V0_MC_PfType1Met'), 
         label  = cms.string('PfType1Met') 
      )
   ) 
) 

process.dbWriterXYshift = cms.EDAnalyzer('METCorrectorDBWriter', 
   #era    = cms.untracked.string('Summer16_V0_DATA_MEtXY'), 
   era    = cms.untracked.string('Summer16_V0_MC_MEtXY'), 
   algo   = cms.untracked.string('PfType1Met'), 
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/')
) 

process.p = cms.Path( 
process.dbWriterXYshift 
) 
