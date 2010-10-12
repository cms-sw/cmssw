import FWCore.ParameterSet.Config as cms 
process = cms.Process('jecdb') 
process.load('CondCore.DBCommon.CondDBCommon_cfi') 
process.CondDBCommon.connect = 'sqlite_file:JEC_Spring10.db' 
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1)) 
process.source = cms.Source('EmptySource') 
process.PoolDBOutputService = cms.Service('PoolDBOutputService', 
   process.CondDBCommon, 
   toPut = cms.VPSet( 
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_AK5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK5Calo'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_AK5Calo') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5Calo'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5Calo') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_AK5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK5PF'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_AK5PF') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5PF'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5PF') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_AK5JPT'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK5JPT'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_AK5JPT') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Summer10_AK5JPT'), 
         tag    = cms.string('JetCorrectorParametersCollection_Summer10_AK5JPT'), 
         label  = cms.string('JetCorrectorParametersCollection_Summer10_AK5JPT') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5JPT'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5JPT'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10DataV2_AK5JPT') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_AK5TRK'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK5TRK'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_AK5TRK') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_AK7Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK7Calo'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_AK7Calo') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_AK7PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_AK7PF'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_AK7PF') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_IC5Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_IC5Calo'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_IC5Calo') 
      ),
     cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_IC5PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_IC5PF'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_IC5PF') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_KT4Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_KT4Calo'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_KT4Calo') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_KT4PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_KT4PF'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_KT4PF') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_KT6Calo'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_KT6Calo'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_KT6Calo') 
      ),
      cms.PSet(
         record = cms.string('JetCorrectorParametersCollection_Spring10_KT6PF'), 
         tag    = cms.string('JetCorrectorParametersCollection_Spring10_KT6PF'), 
         label  = cms.string('JetCorrectorParametersCollection_Spring10_KT6PF') 
      )
   ) 
) 

process.dbWriterAK5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'), 
   algo   = cms.untracked.string('AK5Calo') 
) 
process.dbWriterAK5CaloDataV2 = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10DataV2'),  
   algo   = cms.untracked.string('AK5Calo') 
) 
process.dbWriterAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('AK5PF') 
) 
process.dbWriterAK5PFDataV2 = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10DataV2'), 
   algo   = cms.untracked.string('AK5PF') 
)
process.dbWriterAK5JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo  = cms.untracked.string('AK5JPT') 
)
process.dbWriterAK5JPTSUmmer10 = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Summer10'),  
   algo  = cms.untracked.string('AK5JPT') 
)
process.dbWriterAK5JPTDataV2 = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10DataV2'),  
   algo   = cms.untracked.string('AK5JPT') 
)
process.dbWriterAK5TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'), 
   algo   = cms.untracked.string('AK5TRK') 
)
process.dbWriterAK7Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('AK7Calo') 
) 
process.dbWriterAK7PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('AK7PF') 
) 
process.dbWriterKT4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('KT4Calo') 
) 
process.dbWriterKT4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('KT4PF') 
) 
process.dbWriterKT6Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'), 
   algo   = cms.untracked.string('KT6Calo') 
) 
process.dbWriterKT6PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('KT6PF') 
)
process.dbWriterIC5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('IC5Calo') 
) 
process.dbWriterIC5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   era    = cms.untracked.string('Spring10'),  
   algo   = cms.untracked.string('IC5PF') 
) 



process.p = cms.Path( 
process.dbWriterAK5Calo *
process.dbWriterAK5CaloDataV2 *
process.dbWriterAK5PF *
process.dbWriterAK5PFDataV2 *
process.dbWriterAK5JPT *
process.dbWriterAK5JPTSUmmer10 *
process.dbWriterAK5JPTDataV2 *
process.dbWriterAK5TRK *
process.dbWriterAK7Calo *
process.dbWriterAK7PF *
process.dbWriterKT4Calo *
process.dbWriterKT4PF *
process.dbWriterKT6Calo *
process.dbWriterKT6PF *
process.dbWriterIC5Calo *
process.dbWriterIC5PF
) 
