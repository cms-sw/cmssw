import FWCore.ParameterSet.Config as cms 
process = cms.Process('metdb') 
process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDBCommon.connect = 'sqlite_file:MET15V0.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         record = cms.string('PFMET'), 
         tag    = cms.string('METCorrectorParametersCollection_MET15V0'), 
         label  = cms.string('PFMET') 
      )
   ) 
) 

process.dbWriterXYshift = cms.EDAnalyzer('METCorrectorDBWriter', 
   era    = cms.untracked.string('MET15V0'), 
   algo   = cms.untracked.string('PFMET') 
) 

process.p = cms.Path( 
process.dbWriterXYshift 
) 
