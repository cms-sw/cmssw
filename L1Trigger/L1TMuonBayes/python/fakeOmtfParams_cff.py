import FWCore.ParameterSet.Config as cms

omtfParamsSource = cms.ESSource(
    "EmptyESSource",
    recordName = cms.string('L1TMuonOverlapParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

###OMTF ESProducer. Fills CondFormats from XML files.
omtfParams = cms.ESProducer(
    "L1TMuonBayesOmtfParamsESProducer",
    patternsXMLFiles = cms.VPSet(
        cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/Patterns_0x0003.xml")),
        #cms.PSet(patternsXMLFile = cms.FileInPath("L1Trigger/L1TMuonBayes/test/expert/Patterns_0x0003_TT.xml")),
    ),
    configXMLFile = cms.FileInPath("L1Trigger/L1TMuon/data/omtf_config/hwToLogicLayer_0x0006.xml"),
)



