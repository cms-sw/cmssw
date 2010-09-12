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
         record = cms.string('AK5Calo'), 
         tag    = cms.string('AK5Calo'), 
         label  = cms.string('AK5Calo') 
      ),
      cms.PSet(
         record = cms.string('AK5PF'), 
         tag    = cms.string('AK5PF'), 
         label  = cms.string('AK5PF') 
      ),
      cms.PSet(
         record = cms.string('AK5JPT'), 
         tag    = cms.string('AK5JPT'), 
         label  = cms.string('AK5JPT') 
      ),
      cms.PSet(
         record = cms.string('AK5TRK'), 
         tag    = cms.string('AK5TRK'), 
         label  = cms.string('AK5TRK') 
      ),
      cms.PSet(
         record = cms.string('AK7Calo'), 
         tag    = cms.string('AK7Calo'), 
         label  = cms.string('AK7Calo') 
      ),
      cms.PSet(
         record = cms.string('AK7PF'), 
         tag    = cms.string('AK7PF'), 
         label  = cms.string('AK7PF') 
      ),
      cms.PSet(
         record = cms.string('IC5Calo'), 
         tag    = cms.string('IC5Calo'), 
         label  = cms.string('IC5Calo') 
      ),
     cms.PSet(
         record = cms.string('IC5PF'), 
         tag    = cms.string('IC5PF'), 
         label  = cms.string('IC5PF') 
      ),
      cms.PSet(
         record = cms.string('KT4Calo'), 
         tag    = cms.string('KT4Calo'), 
         label  = cms.string('KT4Calo') 
      ),
      cms.PSet(
         record = cms.string('KT4PF'), 
         tag    = cms.string('KT4PF'), 
         label  = cms.string('KT4PF') 
      ),
      cms.PSet(
         record = cms.string('KT6Calo'), 
         tag    = cms.string('KT6Calo'), 
         label  = cms.string('KT6Calo') 
      ),
      cms.PSet(
         record = cms.string('KT6PF'), 
         tag    = cms.string('KT6PF'), 
         label  = cms.string('KT6PF') 
      )
   ) 
) 

process.dbWriterAK5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('AK5Calo') 
) 
process.dbWriterAK5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('AK5PF') 
) 
process.dbWriterAK5JPT = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('AK5JPT') 
)
process.dbWriterAK5TRK = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('AK5TRK') 
)
process.dbWriterAK7Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('AK7Calo') 
) 
process.dbWriterAK7PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('AK7PF') 
) 
process.dbWriterKT4Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('KT4Calo') 
) 
process.dbWriterKT4PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('KT4PF') 
) 
process.dbWriterKT6Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('KT6Calo') 
) 
process.dbWriterKT6PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('KT6PF') 
)
process.dbWriterIC5Calo = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('IC5Calo') 
) 
process.dbWriterIC5PF = cms.EDAnalyzer('JetCorrectorDBWriter', 
   inputTxtFile = cms.untracked.string('Spring10'), 
   option       = cms.untracked.string(''), 
   label        = cms.untracked.string('IC5PF') 
) 



process.p = cms.Path( 
process.dbWriterAK5Calo *
process.dbWriterAK5PF *
process.dbWriterAK5JPT *
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
