import FWCore.ParameterSet.Config as cms 
process = cms.Process('metdb') 
process.load('CondCore.CondDB.CondDB_cfi') 
#process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDB.connect = 'sqlite_file:MET16V0.db' 
#process.CondDBCommon.connect = 'sqlite_file:MET15V0.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDB, 
   #process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         #record = cms.string('METCorrectionsRecord'), 
         record = cms.string('PfType1Met'), 
         tag    = cms.string('METCorrectorParametersCollection_MET16V0'), 
         label  = cms.string('PfType1Met') 
      )
   ) 
) 

process.dbWriterXYshift = cms.EDAnalyzer('METCorrectorDBWriter', 
   era    = cms.untracked.string('MET16V0'), 
   algo   = cms.untracked.string('PfType1Met') 
) 

process.p = cms.Path( 
process.dbWriterXYshift 
) 
