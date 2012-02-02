import FWCore.ParameterSet.Config as cms 
process = cms.Process('jecdb') 
process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDBCommon.connect = 'sqlite_file:Jec11_V3.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         record = cms.string('AK5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK5Calo'), 
         label  = cms.string('AK5Calo') 
      ),
      cms.PSet(
         record = cms.string('AK5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK5PF'), 
         label  = cms.string('AK5PF') 
      ),
      cms.PSet(
         record = cms.string('AK5PFchs'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK5PFchs'), 
         label  = cms.string('AK5PFchs') 
      ),
      cms.PSet(
         record = cms.string('AK5JPT'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK5JPT'), 
         label  = cms.string('AK5JPT') 
      ),
      cms.PSet(
         record = cms.string('AK5TRK'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK5TRK'), 
         label  = cms.string('AK5TRK') 
      ),
      cms.PSet(
         record = cms.string('AK7Calo'),
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK7Calo'), 
         label  = cms.string('AK7Calo') 
      ),
      cms.PSet(
         record = cms.string('AK7PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK7PF'), 
         label  = cms.string('AK7PF') 
      ),
      cms.PSet(
         record = cms.string('AK7JPT'),                           
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_AK7JPT'),
         label  = cms.string('AK7JPT')                                          
      ),
      cms.PSet(
         record = cms.string('IC5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_IC5Calo'), 
         label  = cms.string('IC5Calo') 
      ),
     cms.PSet(
         record = cms.string('IC5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_IC5PF'), 
         label  = cms.string('IC5PF') 
      ),
      cms.PSet(
         record = cms.string('KT4Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_KT4Calo'), 
         label  = cms.string('KT4Calo') 
      ),
      cms.PSet(
         record = cms.string('KT4PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_KT4PF'), 
         label  = cms.string('KT4PF') 
      ),
      cms.PSet(
         record = cms.string('KT6Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_KT6Calo'), 
         label  = cms.string('KT6Calo') 
      ),
      cms.PSet(
         record = cms.string('KT6PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V3_KT6PF'), 
         label  = cms.string('KT6PF') 
      )
   ) 
) 

process.dbWriterAK5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'), 
   algo   = cms.untracked.string('AK5Calo') 
) 
process.dbWriterAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('AK5PF') 
) 
process.dbWriterAK5PFchs = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('AK5PFchs') 
) 
process.dbWriterAK5JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo  = cms.untracked.string('AK5JPT') 
)
process.dbWriterAK5TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'), 
   algo   = cms.untracked.string('AK5TRK') 
)
process.dbWriterAK7Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('AK7Calo') 
) 
process.dbWriterAK7PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('AK7PF') 
) 
process.dbWriterAK7JPT = cms.EDAnalyzer('JetCorrectorDBWriter',
   era    = cms.untracked.string('Jec11_V3'),
   algo   = cms.untracked.string('AK7JPT')
)
process.dbWriterKT4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('KT4Calo') 
) 
process.dbWriterKT4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('KT4PF') 
) 
process.dbWriterKT6Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'), 
   algo   = cms.untracked.string('KT6Calo') 
) 
process.dbWriterKT6PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('KT6PF') 
)
process.dbWriterIC5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('IC5Calo') 
) 
process.dbWriterIC5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V3'),  
   algo   = cms.untracked.string('IC5PF') 
) 



process.p = cms.Path( 
process.dbWriterAK5Calo *
process.dbWriterAK5PF *
process.dbWriterAK5PFchs *
process.dbWriterAK5JPT *
process.dbWriterAK5TRK *
process.dbWriterAK7Calo *
process.dbWriterAK7PF *
process.dbWriterAK7JPT *
process.dbWriterKT4Calo *
process.dbWriterKT4PF *
process.dbWriterKT6Calo *
process.dbWriterKT6PF *
process.dbWriterIC5Calo *
process.dbWriterIC5PF
) 
