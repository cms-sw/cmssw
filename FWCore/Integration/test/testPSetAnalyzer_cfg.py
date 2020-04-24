import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.demo = cms.EDAnalyzer('TestPSetAnalyzer',
    testLumi = cms.LuminosityBlockID("1:2"),
    testVLumi = cms.VLuminosityBlockID("1:2","2:2","3:3"),
    testRange = cms.LuminosityBlockRange("1:2-4:MAX"),
    testVRange = cms.VLuminosityBlockRange("1:2-4:MAX","99:99","3:4-5:9"),
    testERange = cms.EventRange("1:2-4:MAX"),
    testVERange = cms.VEventRange("1:2-4:MAX","99:99","3:4-5:9","9:9-11:MIN"),
    testEvent = cms.LuminosityBlockID("1:2"),
    testEventID1 = cms.EventID(1, 1, 1000123456789),
    testEventID2 = cms.EventID(2, 1000123456790),
    testEventID3 = cms.EventID('3:3:1000123456791'),
    testEventID4 = cms.EventID('4:1000123456792'),
    testVEventID = cms.VEventID('1:1:1000123456789', '2:1000123456790', cms.EventID(3, 3, 1000123456791)),
    testERange1 = cms.EventRange("1:2:1000123456789-2:3:1000123456790"),
    testERange2 = cms.EventRange("3:1000123456791-4:1000123456792"),
    testERange3 = cms.EventRange("5:6:1000123456793"),
    testERange4 = cms.EventRange(7,8,1000123456794,9,10, 1000123456795),
    testERange5 = cms.EventRange(11, 1000123456796, 12, 1000123456797),
    testVERange2 = cms.VEventRange("1:2:1000123456789-2:3:1000123456790", "1:2-4:1000123456789",cms.EventRange("3:1000123456791-4:1000123456792"),cms.EventRange(11, 1000123456796, 12, 1000123456797))
)

process.p = cms.Path(process.demo)
