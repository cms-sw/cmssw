# Useful for HCAL validation - coherent with Validation/CaloTowers/test/runNoise_NZS_cfg.py
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process('TEST')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('FastSimulation.Configuration.EventContent_cff')
process.load('FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff')
process.load('FastSimulation.Configuration.Geometries_MC_cff')
process.load("Configuration.StandardSequences.MagneticField_0T_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('FastSimulation.Configuration.FamosSequences_cff')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load('FastSimulation.Configuration.HLT_GRun_cff')
process.load('FastSimulation.Configuration.Validation_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Input source
process.source = cms.Source("PoolSource",
                            firstEvent = cms.untracked.uint32(1),
                            noEventSort = cms.untracked.bool(True),	
                            duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            fileNames = cms.untracked.vstring(
        'file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/3_1_X/mc_nue.root'
        )
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )

process.options = cms.untracked.PSet(

    )


# Output definition

process.FEVT = cms.OutputModule("PoolOutputModule",
                                outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
                                fileName = cms.untracked.string("HcalValHarvestingEDM.root")
                                )

# DQM

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

#-----------------------------------------------------------------------------
#                     Ananlyser + slient -> DQM
#-----------------------------------------------------------------------------
process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow


# Make the tracker transparent (unuseful?)
process.famosSimHits.MaterialEffects.PairProduction = False
process.famosSimHits.MaterialEffects.Bremsstrahlung = False
process.famosSimHits.MaterialEffects.EnergyLoss = False
process.famosSimHits.MaterialEffects.MultipleScattering = False
process.famosSimHits.MaterialEffects.NuclearInteraction = False

# Other statements
process.famosSimHits.SimulateCalorimetry = True
process.famosSimHits.SimulateTracking = False
process.simulation = cms.Sequence(process.simulationWithFamos)
process.HLTEndSequence = cms.Sequence(process.reconstructionWithFamos)

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']

# HCAL validation

process.hcalRecoAnalyzer = cms.EDAnalyzer("HcalRecHitsValidation",
                                          outputFile                = cms.untracked.string('HcalRecHitValidationRelVal.root'),
                                          HBHERecHitCollectionLabel = cms.untracked.InputTag("hbhereco"),
                                          HFRecHitCollectionLabel   = cms.untracked.InputTag("hfreco"),
                                          HORecHitCollectionLabel   = cms.untracked.InputTag("horeco"),
                                          eventype                  = cms.untracked.string('single'),
                                          mc                        = cms.untracked.string('yes'),
                                          sign = cms.untracked.string('*'),
                                          hcalselector              = cms.untracked.string('noise'),
                                          ecalselector              = cms.untracked.string('no'),
                                          useAllHistos              = cms.untracked.bool(True),
                                          Famos                     = cms.untracked.bool(True) 
                                          )

process.hcalrechitsClient = cms.EDAnalyzer("HcalRecHitsClient", 
                                           outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
                                           DQMDirName = cms.string("/") # root directory
                                           )


# Path and EndPath definitions
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVT)
process.hcalHitsValidation_step = cms.EndPath(
    process.hcalRecoAnalyzer *
    process.hcalrechitsClient * 
    process.dqmSaver)
# process.MEtoEDMConverter
# )



# Schedule definition
process.schedule = cms.Schedule(process.HLTSchedule)
process.schedule.extend([process.hcalHitsValidation_step,process.FEVTDEBUGHLToutput_step])

