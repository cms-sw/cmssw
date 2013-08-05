import FWCore.ParameterSet.Config as cms

XMLFromDBSource = cms.ESProducer("XMLIdealMagneticFieldGeometryESProducer",
                                 rootDDName = cms.string('cmsMagneticField:MAGF'),
                                 label = cms.string('magfield')
                                 )
