import FWCore.ParameterSet.Config as cms

XMLFromDBSource = cms.ESProducer("XMLIdealGeometryESProducer",
                                 rootDDName = cms.string('cms:OCMS')
                                 )
