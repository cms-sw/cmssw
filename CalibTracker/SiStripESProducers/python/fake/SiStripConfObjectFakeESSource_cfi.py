import FWCore.ParameterSet.Config as cms

siStripConfObjectFakeESSource = cms.ESSource("SiStripConfObjectFakeESSource",
                                         appendToDataLabel = cms.string(''),
                                         Parameters = cms.VPSet(
                                                cms.PSet(
                                                    ParameterName = cms.string("par1"),
                                                    ParameterType = cms.string("int"),
                                                    ParameterValue = cms.int32(1),
                                                ),
                                                cms.PSet(
                                                    ParameterName = cms.string("par2"),
                                                    ParameterType = cms.string("double"),
                                                    ParameterValue = cms.double(1.1),
                                                ),
                                                cms.PSet(
                                                    ParameterName = cms.string("par3"),
                                                    ParameterType = cms.string("string"),
                                                    ParameterValue = cms.string("one"),
                                                ),
                                            )
                                         )



