import FWCore.ParameterSet.Config as cms 
process = cms.Process('jecdb') 
process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDBCommon.connect = 'sqlite_file:Jec11_V10.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         record = cms.string('AK4Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4Calo'), 
         label  = cms.string('AK4Calo') 
      ),
      cms.PSet(
         record = cms.string('AK4PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4PF'), 
         label  = cms.string('AK4PF') 
      ),
      cms.PSet(
         record = cms.string('AK4PFchs'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4PFchs'), 
         label  = cms.string('AK4PFchs') 
      ),
      cms.PSet(
         record = cms.string('AK4JPT'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4JPT'), 
         label  = cms.string('AK4JPT') 
      ),
      cms.PSet(
         record = cms.string('AK4TRK'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_AK4TRK'), 
         label  = cms.string('AK4TRK') 
      ),
      cms.PSet(
         record = cms.string('AK8Calo'),
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK8Calo'), 
         label  = cms.string('AK8Calo') 
      ),
      cms.PSet(
         record = cms.string('AK8PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK8PF'), 
         label  = cms.string('AK8PF') 
      ),
      cms.PSet(
         record = cms.string('AK8JPT'),                           
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK8JPT'),
         label  = cms.string('AK8JPT')                                          
      ),
      cms.PSet(
         record = cms.string('AK4Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_AK4Calo'), 
         label  = cms.string('AK4Calo') 
      ),
     cms.PSet(
         record = cms.string('AK4PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_AK4PF'), 
         label  = cms.string('AK4PF') 
      ),
      cms.PSet(
         record = cms.string('KT4Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_KT4Calo'), 
         label  = cms.string('KT4Calo') 
      ),
      cms.PSet(
         record = cms.string('KT4PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_KT4PF'), 
         label  = cms.string('KT4PF') 
      ),
      cms.PSet(
         record = cms.string('KT6Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_KT6Calo'), 
         label  = cms.string('KT6Calo') 
      ),
      cms.PSet(
         record = cms.string('KT6PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_KT6PF'), 
         label  = cms.string('KT6PF') 
      )
   ) 
) 

process.dbWriterAK4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'), 
   algo   = cms.untracked.string('AK4Calo') 
) 
process.dbWriterAK4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK4PF') 
) 
process.dbWriterAK4PFchs = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK4PFchs') 
) 
process.dbWriterAK4JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo  = cms.untracked.string('AK4JPT') 
)
process.dbWriterAK4TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'), 
   algo   = cms.untracked.string('AK4TRK') 
)
process.dbWriterAK8Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK8Calo') 
) 
process.dbWriterAK8PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK8PF') 
) 
process.dbWriterAK8JPT = cms.EDAnalyzer('JetCorrectorDBWriter',
   era    = cms.untracked.string('Jec11_V5'),
   algo   = cms.untracked.string('AK8JPT')
)
process.dbWriterKT4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('KT4Calo') 
) 
process.dbWriterKT4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('KT4PF') 
) 
process.dbWriterKT6Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'), 
   algo   = cms.untracked.string('KT6Calo') 
) 
process.dbWriterKT6PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('KT6PF') 
)
process.dbWriterAK4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('AK4Calo') 
) 
process.dbWriterAK4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('AK4PF') 
) 



process.p = cms.Path( 
process.dbWriterAK4Calo *
process.dbWriterAK4PF *
process.dbWriterAK4PFchs *
process.dbWriterAK4JPT *
process.dbWriterAK4TRK *
process.dbWriterAK8Calo *
process.dbWriterAK8PF *
process.dbWriterAK8JPT *
process.dbWriterKT4Calo *
process.dbWriterKT4PF *
process.dbWriterKT6Calo *
process.dbWriterKT6PF *
process.dbWriterAK4Calo *
process.dbWriterAK4PF
) 
