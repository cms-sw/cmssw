import FWCore.ParameterSet.Config as cms 
process = cms.Process('metdb') 
process.load('CondCore.CondDB.CondDB_cfi') 
#process.load('CondCore.DBCommon.CondDBCommon_cfi') 
#process.CondDB.connect = 'sqlite_file:Spring16_V0_MET_Data.db' 
process.CondDB.connect = 'sqlite_file:Spring16_V0_MET_MC.db' 
#process.CondDBCommon.connect = 'sqlite_file:MET15V0.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDB, 
   #process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         record = cms.string('PfType1Met'), 
         #tag    = cms.string('METCorrectorParametersCollection_Spring16_V0_Data_PfType1Met'), 
         tag    = cms.string('METCorrectorParametersCollection_Spring16_V0_MC_PfType1Met'), 
         label  = cms.string('PfType1Met') 
      )
   ) 
) 

process.dbWriterXYshift = cms.EDAnalyzer('METCorrectorDBWriter', 
   #era    = cms.untracked.string('Spring16_V0_MET_Data'), 
   era    = cms.untracked.string('Spring16_V0_MET_MC'), 
   algo   = cms.untracked.string('PfType1Met'), 
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/')
) 

process.p = cms.Path( 
process.dbWriterXYshift 
) 
