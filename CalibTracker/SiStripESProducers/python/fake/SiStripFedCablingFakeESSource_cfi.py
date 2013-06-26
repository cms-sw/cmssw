import FWCore.ParameterSet.Config as cms

siStripFedCabling = cms.ESSource("SiStripFedCablingFakeESSource",
                                 FedIdsFile       = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripFedIdList.dat'),
                                 DetIdsFile       = cms.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
                                 PopulateAllFeds  = cms.bool(False)
                                 )


