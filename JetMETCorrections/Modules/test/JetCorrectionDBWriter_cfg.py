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
         record = cms.string('AK7Calo'),
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK7Calo'), 
         label  = cms.string('AK7Calo') 
      ),
      cms.PSet(
         record = cms.string('AK7PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK7PF'), 
         label  = cms.string('AK7PF') 
      ),
      cms.PSet(
         record = cms.string('AK7JPT'),                           
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK7JPT'),
         label  = cms.string('AK7JPT')                                          
      ),
      cms.PSet(
         record = cms.string('IC5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_IC5Calo'), 
         label  = cms.string('IC5Calo') 
      ),
     cms.PSet(
         record = cms.string('IC5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Jec11_V5_IC5PF'), 
         label  = cms.string('IC5PF') 
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
   algo   = cms.untracked.string('AK5Calo'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK5PF'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterAK5PFchs = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK5PFchs'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterAK5JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo  = cms.untracked.string('AK5JPT'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
)
process.dbWriterAK5TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'), 
   algo   = cms.untracked.string('AK5TRK'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
)
process.dbWriterAK7Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK7Calo'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterAK7PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('AK7PF'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterAK7JPT = cms.EDAnalyzer('JetCorrectorDBWriter',
   era    = cms.untracked.string('Jec11_V5'),
   algo   = cms.untracked.string('AK7JPT'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
)
process.dbWriterKT4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('KT4Calo'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterKT4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V10'),  
   algo   = cms.untracked.string('KT4PF'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterKT6Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'), 
   algo   = cms.untracked.string('KT6Calo'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterKT6PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('KT6PF'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
)
process.dbWriterIC5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('IC5Calo'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
) 
process.dbWriterIC5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Jec11_V5'),  
   algo   = cms.untracked.string('IC5PF'),
   path   = cms.untracked.string('CondFormats/JetMETObjects/data/'),
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
