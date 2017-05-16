import FWCore.ParameterSet.Config as cms

emtfParamsSource = cms.ESSource(
	"EmptyESSource",
	recordName = cms.string('L1TMuonEndcapParamsRcd'),
	iovIsRunNotTime = cms.bool(True),
	firstValid = cms.vuint32(1)
)

##EMTF ESProducer. Fills CondFormats from XML files.
emtfParams = cms.ESProducer(
	"L1TMuonEndCapParamsESProducer",
   PtAssignVersion = cms.int32(4),
   firmwareVersion = cms.int32(50),
   pcLutVersion    = cms.int32(1)
)



emtfForestsSource = cms.ESSource(
	"EmptyESSource",
	recordName = cms.string('L1TMuonEndCapForestRcd'),
	iovIsRunNotTime = cms.bool(True),
	firstValid = cms.vuint32(1)
)

##EMTF ESProducer. Fills CondFormats from XML files.
emtfForests = cms.ESProducer(
	"L1TMuonEndCapForestESProducer",
)




