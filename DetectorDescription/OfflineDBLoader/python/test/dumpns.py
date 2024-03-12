import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
                                        geomXMLFiles = cms.vstring('DetectorDescription/OfflineDBLoader/test/fred.xml'),
                                        rootNodeName = cms.string('cms:OCMS')
                                        )

# foo bar baz
# 3Fw1jXWD7KokM
# phnbBu2rEiKRb
