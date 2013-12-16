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
         record = cms.string('AK5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK5Calo'), 
         label  = cms.string('AK5Calo') 
      ),
      cms.PSet(
         record = cms.string('AK5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK5PF'), 
         label  = cms.string('AK5PF') 
      ),
      cms.PSet(
         record = cms.string('AK5PFchs'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK5PFchs'), 
         label  = cms.string('AK5PFchs') 
      ),
      cms.PSet(
         record = cms.string('AK5JPT'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK5JPT'), 
         label  = cms.string('AK5JPT') 
      ),
      cms.PSet(
         record = cms.string('AK5TRK'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_AK5TRK'), 
         label  = cms.string('AK5TRK') 
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
         record = cms.string('AK5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_AK5Calo'), 
         label  = cms.string('AK5Calo') 
      ),
     cms.PSet(
         record = cms.string('AK5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_AK5PF'), 
         label  = cms.string('AK5PF') 
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

process.dbWriterAK5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'), 
   algo   = cms.untracked.string('AK5Calo') 
) 
process.dbWriterAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK5PF') 
) 
process.dbWriterAK5PFchs = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK5PFchs') 
) 
process.dbWriterAK5JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo  = cms.untracked.string('AK5JPT') 
)
process.dbWriterAK5TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'), 
   algo   = cms.untracked.string('AK5TRK') 
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
process.dbWriterAK5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('AK5Calo') 
) 
process.dbWriterAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('AK5PF') 
) 



process.p = cms.Path( 
process.dbWriterAK5Calo *
process.dbWriterAK5PF *
process.dbWriterAK5PFchs *
process.dbWriterAK5JPT *
process.dbWriterAK5TRK *
process.dbWriterAK8Calo *
process.dbWriterAK8PF *
process.dbWriterAK8JPT *
process.dbWriterKT4Calo *
process.dbWriterKT4PF *
process.dbWriterKT6Calo *
process.dbWriterKT6PF *
process.dbWriterAK5Calo *
process.dbWriterAK5PF
) 
