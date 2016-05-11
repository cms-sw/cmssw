
import FWCore.ParameterSet.Config as cms

process = cms.Process("metdbreader")
process.load('Configuration.StandardSequences.Services_cff')
#process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load("CondCore.CondDB.CondDB_cfi")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
      DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0)
        ),
      timetype = cms.string('runnumber'),
      toGet = cms.VPSet(
      cms.PSet(
              #record = cms.string('MetShiftXY'),
              #record = cms.string('PfType1Met'), 
              record = cms.string('METCorrectionsRecord'), 
              #tag    = cms.string('metShiftxy'),
              tag    = cms.string('METCorrectorParametersCollection_MET16V0'),
              label  = cms.untracked.string('PfType1Met')
              #label  = cms.untracked.string('PFMETLocal')
              #label  = cms.untracked.string('AK5CaloLocal') 
            ),                                                                               
       ),
      connect = cms.string('sqlite:MET16V0.db')
      #connect = cms.string('sqlite:MET12_V0.db')
)


process.demo1 = cms.EDAnalyzer('METCorrectorDBReader', 
        payloadName     = cms.untracked.string('PfType1Met'),
        #payloadName    = cms.untracked.string('PFMETLocal'),
        #payloadName    = cms.untracked.string('MetShiftXY'),
        #payloadName    = cms.untracked.string('AK5CaloLocal'),
        printScreen    = cms.untracked.bool(True),
        createTextFile = cms.untracked.bool(True),
        globalTag      = cms.untracked.string('MET16V0')
)


process.p = cms.Path(process.demo1 )
