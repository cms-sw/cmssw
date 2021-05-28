import FWCore.ParameterSet.Config as cms

process = cms.Process('ALCAHARVEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.AlCaHarvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(325022),
                            numberEventsInRun = cms.untracked.uint32(1),
                            numberEventsInLuminosityBlock = cms.untracked.uint32(1),
                            firstTime = cms.untracked.uint64(6614916085915320320),
                            timeBetweenEvents = cms.untracked.uint64(1)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripQuality_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripGains_dbOutput)
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiStripGainsAAG_dbOutput )
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiPixelAli_dbOutput)
process.PoolDBOutputService.toPut.extend(process.ALCAHARVESTSiPixelQuality_dbOutput)

process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripQuality_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripGains_metadata )
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiStripGainsAAG_metadata)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiPixelAli_metadata)
process.pclMetadataWriter.recordsToMap.extend(process.ALCAHARVESTSiPixelQuality_metadata)

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.SiStripQuality  = cms.Path(process.ALCAHARVESTSiStripQuality)
process.alcaSiStripQualityHarvester.CalibrationThreshold = cms.untracked.uint32(0)

process.SiStripGains    = cms.Path(process.ALCAHARVESTSiStripGains)
#process.alcaSiStripGainsHarvester.

process.SiStripGainsAAG = cms.Path(process.ALCAHARVESTSiStripGainsAAG)
#process.alcaSiStripGainsAAGHarvester.

process.SiPixelAli      = cms.Path(process.ALCAHARVESTSiPixelAli)

process.SiPixelQuality  = cms.Path(process.ALCAHARVESTSiPixelQuality)

process.ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(process.dqmSaver+process.pclMetadataWriter)

process.schedule = cms.Schedule(process.SiStripQuality,
                                process.SiStripGains,    
                                process.SiStripGainsAAG, 
                                process.SiPixelAli,      
                                process.SiPixelQuality,
                                process.ALCAHARVESTDQMSaveAndMetadataWriter)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
