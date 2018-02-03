import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1TdeCSCTF = DQMEDAnalyzer('L1TdeCSCTF',
	outFile=cms.untracked.string(""),
	DQMStore=cms.untracked.bool(True),
	DQMFolder=cms.untracked.string("L1TEMU/CSCTFexpert"),
	disableROOToutput=cms.untracked.bool(True),
	dataTrackProducer=cms.InputTag("csctfDigis"),
	emulTrackProducer=cms.InputTag("valCsctfTrackDigis"),
	lctProducer=cms.InputTag("csctfDigis"),
	dataStubProducer=cms.InputTag("csctfDigis","DT"),
	emulStubProducer=cms.InputTag("dttfDigis"),
	PTLUT=cms.PSet(
		LowQualityFlag=cms.untracked.uint32(4),
		ReadPtLUT=cms.bool(False),
		PtMethod=cms.untracked.uint32(1),
	)
)

