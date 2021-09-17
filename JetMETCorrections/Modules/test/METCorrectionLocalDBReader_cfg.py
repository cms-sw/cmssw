
import FWCore.ParameterSet.Config as cms

process = cms.Process("metdbreader")



process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('demo1'),
    files = cms.untracked.PSet(
        myDebugOutputFile = cms.untracked.PSet(
            default = cms.untracked.PSet(
                limit = cms.untracked.int32(-1)
            ),
            threshold = cms.untracked.string('DEBUG')
        )
    )
)

#process.load('Configuration.StandardSequences.Services_cff')
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDB.connect = 'sqlite_file:Summer16_V0_DATA_MEtXY.db'
process.CondDB.connect = 'sqlite_file:Summer16_V0_MC_MEtXY.db'

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")


process.PoolDBESSource = cms.ESSource("PoolDBESSource",
      process.CondDB,
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
      cms.PSet(
              record = cms.string('MEtXYcorrectRecord'),# plugin 
              #tag    = cms.string('MEtXYcorrectParametersCollection_Summer16_V0_DATA_PfType1Met'), 
              tag    = cms.string('MEtXYcorrectParametersCollection_Summer16_V0_MC_PfType1Met'),
              #label  = cms.untracked.string('PfType1Met')
              label  = cms.untracked.string('PfType1MetLocal')
            )                                                                               
       )
)


process.demo1 = cms.EDAnalyzer('METCorrectorDBReader', 
        payloadName     = cms.untracked.string('PfType1MetLocal'),
        printScreen    = cms.untracked.bool(True),
        createTextFile = cms.untracked.bool(True),
        #globalTag      = cms.untracked.string('Summer16_V0_DATA_MEtXY')
        globalTag      = cms.untracked.string('Summer16_V0_MC_MEtXY')
)

process.p = cms.Path(process.demo1 )
