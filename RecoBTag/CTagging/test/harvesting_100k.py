import FWCore.ParameterSet.Config as cms

process = cms.Process("harvesting")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.MessageLogger.cerr.threshold = 'ERROR'

# for the conditions
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

# Open and read list file
#filename = open('RootFiles/list.list', 'r')
#filelist = cms.untracked.vstring( filename.readlines() )

# Input source
process.source = cms.Source("PoolSource",
  #fileNames = filelist,
  fileNames = cms.untracked.vstring(),
  secondaryFileNames = cms.untracked.vstring(),
  processingMode = cms.untracked.string('RunsAndLumis')
)


process.options = cms.untracked.PSet(
  Rethrow = cms.untracked.vstring('ProductNotFound'),
  fileMode = cms.untracked.string('FULLMERGE')
)

from DQMOffline.RecoB.bTagCommon_cff import*
process.load("DQMOffline.RecoB.bTagCommon_cff"),
process.bTagCommonBlock.finalizePlots = cms.bool(True)
#process.bTagCommonBlock.ptRecJetMin = cms.double(600.0)

###############################################################################################

from Validation.RecoB.bTagAnalysis_cfi import *
#process.load("Validation.RecoB.bTagAnalysis_harvesting_cfi")
process.bTagHarvestMC.ptRanges = cms.vdouble(0.0,40.0,60.0,90.0, 150.0,400.0,600.0,3000.0)
process.bTagHarvestMC.etaRanges = cms.vdouble(0.0, 1.2, 2.1, 2.4)
#process.bTagHarvestMC.doPUid = cms.bool(True)
process.bTagHarvestMC.flavPlots = cms.string("alldusg")

process.bTagHarvestMC.tagConfig += cms.VPSet(
#		cms.PSet(
#            cTagGenericAnalysisBlock,
#            label = cms.InputTag("combinedInclusiveSecondaryVertexV2BJetTags"),
#            folder = cms.string("CSVv2_tkOnly")
	    #doCTagPlots = cms.bool(True)
#		)
	)

process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/cMVA'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.bTagValidation_step = cms.Path(process.bTagHarvestMC)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(
  process.edmtome_step,
  process.bTagValidation_step,
  process.dqmsave_step
)

process.PoolSource.fileNames = [
'file:DQMfile.root'
]
