
import FWCore.ParameterSet.Config as cms

process = cms.Process("metdbreader")
#process.load('Configuration.StandardSequences.Services_cff')
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDB.connect = 'sqlite_file:Spring16_V0_MET_Data.db'
process.CondDB.connect = 'sqlite_file:Spring16_V0_MET_MC.db'

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")


process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring(
      'myDebugOutputFile'
      ),
    myDebugOutputFile = cms.untracked.PSet(
      threshold = cms.untracked.string('DEBUG'),
      default = cms.untracked.PSet(
	limit = cms.untracked.int32(-1)
	),
      ),
    debugModules = cms.untracked.vstring(
      'demo1'
      )
    )

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
      process.CondDB,
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
      cms.PSet(
              record = cms.string('METCorrectionsRecord'),# plugin 
              #tag    = cms.string('METCorrectorParametersCollection_Spring16_V0_Data_PfType1Met'),
              tag    = cms.string('METCorrectorParametersCollection_Spring16_V0_MC_PfType1Met'),
              #label  = cms.untracked.string('PfType1Met')
              label  = cms.untracked.string('PfType1MetLocal')
            )                                                                               
       )
)


process.demo1 = cms.EDAnalyzer('METCorrectorDBReader', 
        payloadName     = cms.untracked.string('PfType1MetLocal'),
        printScreen    = cms.untracked.bool(True),
        createTextFile = cms.untracked.bool(True),
        #globalTag      = cms.untracked.string('Spring16_V0_MET_Data')
        globalTag      = cms.untracked.string('Spring16_V0_MET_MC')
)

process.p = cms.Path(process.demo1 )
