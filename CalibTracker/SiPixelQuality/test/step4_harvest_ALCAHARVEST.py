# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step4_harvest --conditions auto:run2_data -s ALCAHARVEST:SiPixelQuality --data --era Run2_2017 --filein file:PromptCalibProdSiPixel.root -n -1 --customise_commands=process.GlobalTag.toGet.append(cms.PSet(record=cms.string("SiPixelQualityFromDbRcd"), tag=cms.string("SiPixelQuality_v04_offline"), connect=cms.string("frontier://FrontierProd/CMS_CONDITIONS")))\n --no_exec
import FWCore.ParameterSet.Config as cms


from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process('ALCAHARVEST',Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
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
    annotation = cms.untracked.string('step4_harvest nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
process.PoolDBOutputService.toPut.extend(process.ALCAHARVESTSiPixelQuality_dbOutput)
process.pclMetadataWriter.recordsToMap.extend(process.ALCAHARVESTSiPixelQuality_metadata)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

# Path and EndPath definitions
process.BeamSpotHPLowPUByRun = cms.Path(process.ALCAHARVESTBeamSpotHPLowPUByRun)
process.ALCAHARVESTDQMSaveAndMetadataWriter = cms.Path(process.dqmSaver+process.pclMetadataWriter)
process.EcalPedestals = cms.Path(process.ALCAHARVESTEcalPedestals)
process.LumiPCC = cms.Path(process.ALCAHARVESTLumiPCC)
process.BeamSpotByRun = cms.Path(process.ALCAHARVESTBeamSpotByRun)

process.ALCAHARVESTSiPixelQuality.SiPixelStatusManagerParameters.threshold = cms.untracked.double(0.01)
process.SiPixelQuality = cms.Path(process.ALCAHARVESTSiPixelQuality)#+process.siPixelPhase1DQMHarvester)

process.BeamSpotHPLowPUByLumi = cms.Path(process.ALCAHARVESTBeamSpotHPLowPUByLumi)
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

process.GlobalTag.toGet.append(cms.PSet(record=cms.string("SiPixelQualityFromDbRcd"), tag=cms.string("SiPixelQuality_v04_offline"), connect=cms.string("frontier://FrontierProd/CMS_CONDITIONS")))

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
