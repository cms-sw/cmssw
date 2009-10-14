import FWCore.ParameterSet.Config as cms

l1decsctf = cms.EDAnalyzer("CSCTFDataToEmuComparator",
    outFile = cms.untracked.string(""),
	DQMStore = cms.untracked.bool(True),
	disableROOToutput = cms.untracked.bool(True),
    dataTrackProducer = cms.string("csctfDigis"),
	emulTrackProducer = cms.string("simCsctfTrackDigis"),
	lctProducer = cms.string("csctfDigis"),
	PTLUT = cms.PSet(
	    LowQualityFlag = cms.untracked.uint32(4),
	    ReadPtLUT = cms.untracked.bool(False),
	    PtMethod = cms.untracked.uint32(1),
    )
)
#TODO: please add input tags for interfacing with the hardware sequence
