import FWCore.ParameterSet.Config as cms

siStripPedestalsFakeESSource = cms.ESSource("SiStripPedestalsFakeESSource",
                                            appendToDataLabel = cms.string(''),
                                            SiStripDetInfoFile = cms.FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"),
                                            printDebug = cms.untracked.uint32(5),
                                            PedestalsValue = cms.uint32(30)
                                            )
# foo bar baz
# By6D4BDQzpBPr
