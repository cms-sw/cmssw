import FWCore.ParameterSet.Config as cms

siStripPedestalsFakeESSource = cms.ESSource("SiStripPedestalsFakeESSource",
                                            appendToDataLabel = cms.string(''),
                                            #
                                            printDebug = cms.untracked.uint32(5),
                                            PedestalsValue = cms.uint32(30)
                                            )
