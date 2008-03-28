import FWCore.ParameterSet.Config as cms

# define an empty EventSetupRecord for the b/tau algorithms that do not need to access the database
BTagRecord = cms.ESSource("EmptyESSource",
    recordName = cms.string('JetTagComputerRecord'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)


