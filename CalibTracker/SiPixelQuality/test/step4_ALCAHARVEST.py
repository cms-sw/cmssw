# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 --conditions 100X_dataRun2_Express_v2 -s ALCAHARVEST:SiPixelQuality --data --era Run2_2017 --filein file:PromptCalibProdSiPixel.root --scenario pp -n 1 --fileout file:step4.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('ALCAHARVEST',eras.Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.AlCaHarvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 50000

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = cms.untracked.vstring(),
    processingMode = cms.untracked.string('RunsAndLumis')
)

process.source.fileNames.extend(['file:SiPixelCalZeroBias.root'])

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step4 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Other statements
process.PoolDBOutputService.toPut.extend(process.ALCAHARVESTSiPixelQuality_dbOutput)
process.pclMetadataWriter.recordsToMap.extend(process.ALCAHARVESTSiPixelQuality_metadata)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '102X_dataRun2_Express_v1', '')

# Path and EndPath definitions
process.ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(process.dqmSaver+process.pclMetadataWriter)
process.EcalPedestals = cms.Path(process.ALCAHARVESTEcalPedestals)
process.LumiPCC = cms.Path(process.ALCAHARVESTLumiPCC)
process.BeamSpotByRun = cms.Path(process.ALCAHARVESTBeamSpotByRun)
process.SiPixelQuality = cms.Path(process.ALCAHARVESTSiPixelQuality)#+process.siPixelPhase1DQMHarvester)
process.SiStripGains = cms.Path(process.ALCAHARVESTSiStripGains)
process.BeamSpotHPByRun = cms.Path(process.ALCAHARVESTBeamSpotHPByRun)
process.SiPixelAli = cms.Path(process.ALCAHARVESTSiPixelAli)
process.BeamSpotByLumi = cms.Path(process.ALCAHARVESTBeamSpotByLumi)
process.BeamSpotHPByLumi = cms.Path(process.ALCAHARVESTBeamSpotHPByLumi)
process.SiStripGainsAAG = cms.Path(process.ALCAHARVESTSiStripGainsAAG)
process.SiStripQuality = cms.Path(process.ALCAHARVESTSiStripQuality)

# Schedule definition
process.schedule = cms.Schedule(process.SiPixelQuality,process.ALCAHARVESTDQMSaveAndMetadataWriter)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
