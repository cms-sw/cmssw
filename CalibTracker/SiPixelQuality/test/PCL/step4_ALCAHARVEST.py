# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4 --data --conditions 100X_dataRun2_Express_v2 --scenario pp -s ALCAHARVEST:SiPixelQuality --filein file:PromptCalibProdSiPixel.root -n -1 --no_exec
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('ALCAHARVEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.AlCaHarvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:PromptCalibProdSiPixel.root'),
    processingMode = cms.untracked.string('RunsAndLumis'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step4 nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
process.PoolDBOutputService.toPut.append(process.ALCAHARVESTSiPixelQuality_dbOutput)
process.pclMetadataWriter.recordsToMap.append(process.ALCAHARVESTSiPixelQuality_metadata)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '100X_dataRun2_Express_v2', '')

# Path and EndPath definitions
process.SiStripGains = cms.Path(process.ALCAHARVESTSiStripGains)
process.EcalPedestals = cms.Path(process.ALCAHARVESTEcalPedestals)
process.SiStripGainsAAG = cms.Path(process.ALCAHARVESTSiStripGainsAAG)
process.BeamSpotByRun = cms.Path(process.ALCAHARVESTBeamSpotByRun)
process.SiPixelQuality = cms.Path(process.ALCAHARVESTSiPixelQuality)
process.BeamSpotHPByRun = cms.Path(process.ALCAHARVESTBeamSpotHPByRun)
process.SiPixelAli = cms.Path(process.ALCAHARVESTSiPixelAli)
process.BeamSpotByLumi = cms.Path(process.ALCAHARVESTBeamSpotByLumi)
process.BeamSpotHPByLumi = cms.Path(process.ALCAHARVESTBeamSpotHPByLumi)
process.ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(process.dqmSaver+process.pclMetadataWriter)
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
