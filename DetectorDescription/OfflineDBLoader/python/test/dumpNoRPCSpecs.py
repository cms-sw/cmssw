import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
                                        geomXMLFiles = cms.vstring('DetectorDescription/OfflineDBLoader/test/fredNoRPCSpecs.xml'),
                                        rootNodeName = cms.string('cms:OCMS'),
                                        userControlledNamespace = cms.untracked.bool(True)
                                        )

# foo bar baz
# 6JHYaGYh6I2CM
# evUOCzItcQZlv
