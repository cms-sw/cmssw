import FWCore.ParameterSet.Config as cms

siStripFedCabling = cms.ESSource("SiStripFedCablingFakeESSource",
                                 SiStripDetInfoFile = cms.FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"),
                                 FedIdsFile       = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripFedIdList.dat'),
                                 PopulateAllFeds  = cms.bool(False)
                                 )


