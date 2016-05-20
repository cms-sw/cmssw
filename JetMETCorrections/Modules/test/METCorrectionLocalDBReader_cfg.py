
import FWCore.ParameterSet.Config as cms

process = cms.Process("metdbreader")
#process.load('Configuration.StandardSequences.Services_cff')
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:MET16V0.db'

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
      process.CondDB,
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
      cms.PSet(
              #record = cms.string('MetShiftXY'),
              #record = cms.string('PfType1Met'), 
              record = cms.string('METCorrectionsRecord'),# plugin 
              #tag    = cms.string('metShiftxy'),
              tag    = cms.string('METCorrectorParametersCollection_MET16V0'),
              #label  = cms.untracked.string('PfType1Met')
              label  = cms.untracked.string('PfType1MetLocal')
              #label  = cms.untracked.string('AK5CaloLocal') 
            )                                                                               
       )
)


process.demo1 = cms.EDAnalyzer('METCorrectorDBReader', 
        payloadName     = cms.untracked.string('PfType1MetLocal'),
        #payloadName     = cms.untracked.string('PfType1Met'),
        #payloadName    = cms.untracked.string('PFMETLocal'),
        #payloadName    = cms.untracked.string('MetShiftXY'),
        #payloadName    = cms.untracked.string('AK5CaloLocal'),
        printScreen    = cms.untracked.bool(True),
        createTextFile = cms.untracked.bool(True),
        globalTag      = cms.untracked.string('MET16V0')
)


process.p = cms.Path(process.demo1 )
