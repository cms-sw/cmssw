
import FWCore.ParameterSet.Config as cms

process = cms.Process("metdbreader")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

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
              record = cms.string('METCorrectionsRecord'), 
              #tag    = cms.string('metShiftxy'),
              tag    = cms.string('METCorrectorParametersCollection_MET15V0'),
              label  = cms.untracked.string('PFMETLocal')
              #label  = cms.untracked.string('AK5CaloLocal') 
            ),                                                                               
       ),
      connect = cms.string('sqlite:MET15V0.db')
      #connect = cms.string('sqlite:MET12_V0.db')
)


process.demo1 = cms.EDAnalyzer('METCorrectorDBReader', 
        payloadName    = cms.untracked.string('PFMETLocal'),
        #payloadName    = cms.untracked.string('MetShiftXY'),
        #payloadName    = cms.untracked.string('AK5CaloLocal'),
        printScreen    = cms.untracked.bool(True),
        createTextFile = cms.untracked.bool(True),
        globalTag      = cms.untracked.string('MET15V0')
)


process.p = cms.Path(process.demo1 )
