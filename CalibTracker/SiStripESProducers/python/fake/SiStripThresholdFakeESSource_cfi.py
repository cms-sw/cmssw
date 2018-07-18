import FWCore.ParameterSet.Config as cms

siStripThresholdFakeESSource = cms.ESSource("SiStripThresholdFakeESSource",
                                            appendToDataLabel = cms.string(''),
                                            HighTh = cms.double(5.0),
                                            LowTh  = cms.double(2.0),
                                            ClusTh = cms.double(0.0)
                                            )



