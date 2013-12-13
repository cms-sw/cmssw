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
            tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4Calo'),
            label  = cms.untracked.string('AK4CaloLocal')
            ),
      cms.PSet(
            record = cms.string('JetCorrectionsRecord'),
            tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4PF'),
            label  = cms.untracked.string('AK4PFLocal')
            ),
      cms.PSet(
            record = cms.string('JetCorrectionsRecord'),
            tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4PFchs'),
            label  = cms.untracked.string('AK4PFchsLocal')
            ),
      cms.PSet(
            record = cms.string('JetCorrectionsRecord'),
            tag    = cms.string('JetCorrectorParametersCollection_Jec11_V10_AK4JPT'),
            label  = cms.untracked.string('AK4JPTLocal')
            ),                                                                                
       ),
      connect = cms.string('sqlite:Jec11_V10.db')
)


process.demo1 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK4CaloLocal'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True),
        globalTag      = cms.untracked.string('Jec11_V10')
)


process.demo2 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK4PFLocal'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True),
        globalTag      = cms.untracked.string('Jec11_V10')
)

process.demo3 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK4PFchsLocal'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True),
        globalTag      = cms.untracked.string('Jec11_V10')
)

process.demo4 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK4JPTLocal'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True),
        globalTag      = cms.untracked.string('Jec11_V10')                               
)

process.p = cms.Path(process.demo1 * process.demo2 * process.demo3 * process.demo4 )
