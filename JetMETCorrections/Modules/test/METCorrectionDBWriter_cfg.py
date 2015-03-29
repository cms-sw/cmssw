# location /JetMETCorrections/Modules/test
import FWCore.ParameterSet.Config as cms 
process = cms.Process('metdb') 
process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDBCommon.connect = 'sqlite_file:MET11_V0.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         record = cms.string('MetShiftXY'), 
         #record = cms.string('METCorrectionsRecord'), 
         tag    = cms.string('METCorrectorParametersCollection_MET15_V0'), 
         label  = cms.string('MetShiftXY') 
      )
   ) 
) 

process.dbWriterXYshift = cms.EDAnalyzer('METCorrectorDBWriter', 
   era    = cms.untracked.string('MET15_V0'), 
   #era    = cms.untracked.string('myEpoch'), 
   algo   = cms.untracked.string('MetShiftXY') 
) 



process.p = cms.Path( 
process.dbWriterXYshift 
#process.dbWriterAK5PF *
#process.dbWriterAK5PFchs *
#process.dbWriterAK5JPT *
#process.dbWriterAK5TRK *
#process.dbWriterAK7Calo *
#process.dbWriterAK7PF *
#process.dbWriterAK7JPT *
#process.dbWriterKT4Calo *
#process.dbWriterKT4PF *
#process.dbWriterKT6Calo *
#process.dbWriterKT6PF *
#process.dbWriterIC5Calo *
#process.dbWriterIC5PF
) 
