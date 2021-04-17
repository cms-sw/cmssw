import FWCore.ParameterSet.Config as cms

dqmXMLFileGetter=cms.EDAnalyzer("DQMXMLFileEventSetupAnalyzer",
                                labelToGet = cms.string('GenericXML')
                                ) 

