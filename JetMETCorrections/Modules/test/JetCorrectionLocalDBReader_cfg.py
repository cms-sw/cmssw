import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
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
            record = cms.string('JetCorrectionsRecord'),
            tag    = cms.string('JetCorrectorParametersCollection_Spring10_V5_AK5Calo'),
            label  = cms.untracked.string('AK5CaloLocal')
            ),
      cms.PSet(
            record = cms.string('JetCorrectionsRecord'),
            tag    = cms.string('JetCorrectorParametersCollection_Spring10_V5_IC5Calo'),
            label  = cms.untracked.string('IC5CaloLocal')
            ),                                                                                
       ),
      connect = cms.string('sqlite:JEC_Spring10.db')
)


process.demo1 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK5CaloLocal'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(False),
        globalTag      = cms.untracked.string('JEC_Spring10')
)

process.demo2 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('IC5CaloLocal'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(False),
        globalTag      = cms.untracked.string('JEC_Spring10')                               
)

process.p = cms.Path(process.demo1 * process.demo2)
