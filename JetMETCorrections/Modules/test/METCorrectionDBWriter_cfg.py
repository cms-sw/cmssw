
##____________________________________________________________________________||
import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process('metdb')

##____________________________________________________________________________||
process.load('CondCore.DBCommon.CondDBCommon_cfi')
process.CondDBCommon.connect = 'sqlite_file:MET11_V0.db'
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source('EmptySource')

##____________________________________________________________________________||
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

##____________________________________________________________________________||
process.dbWriterXYshift = cms.EDAnalyzer('METCorrectorDBWriter',
   era    = cms.untracked.string('MET15_V0'),
   algo   = cms.untracked.string('MetShiftXY')
)

##____________________________________________________________________________||
process.p = cms.Path(
    process.dbWriterXYshift
)

##____________________________________________________________________________||
