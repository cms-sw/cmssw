# Useful for HCAL validation
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

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
    )

# Input source
process.source = cms.Source("PoolSource",
                            #			        firstEvent = cms.untracked.uint32(XXXXX),
                            firstEvent = cms.untracked.uint32(0),
                            #			        fileNames = cms.untracked.vstring('file:mc.root')
                            fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalHits/data/3_1_X/mc_pi50_eta05.root')
			    )

process.options = cms.untracked.PSet(

    )


# Output definition

process.FEVT = cms.OutputModule("PoolOutputModule",
                                outputCommands = cms.untracked.vstring('drop *', 'keep *_MEtoEDMConverter_*_*'),
                                fileName = cms.untracked.string("output.root")
                                )

# DQM

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# Vertex in (0,0,0)

process.load("RecoVertex.BeamSpotProducer.BeamSpotFakeParameters_cfi")
process.es_prefer_beamspot = cms.ESPrefer("BeamSpotFakeConditions","")
process.BeamSpotFakeConditions.X0=0
process.BeamSpotFakeConditions.Y0=0
process.BeamSpotFakeConditions.Z0=0

process.famosSimHits.VertexGenerator.SigmaX=0
process.famosSimHits.VertexGenerator.SigmaY=0
process.famosSimHits.VertexGenerator.SigmaZ=0  

process.VtxSmeared.SigmaX = 0.00001 # unuseful?
process.VtxSmeared.SigmaY = 0.00001 # unuseful?
process.VtxSmeared.SigmaZ = 0.00001 # unuseful?

# Make the tracker transparent
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
                                          ecalselector              = cms.untracked.string('yes'),
                                          hcalselector              = cms.untracked.string('all'),
                                          mc                        = cms.untracked.string('yes'),
                                          Famos                     = cms.untracked.bool(True) 
                                          )

process.hcalTowerAnalyzer = cms.EDAnalyzer("CaloTowersValidation",
                                           outputFile               = cms.untracked.string('CaloTowersValidationRelVal.root'),
                                           CaloTowerCollectionLabel = cms.untracked.InputTag('towerMaker'),
                                           hcalselector             = cms.untracked.string('all'),
                                           mc                       = cms.untracked.string('yes')  
                                           )


# Path and EndPath definitions
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVT)
process.hcalHitsValidation_step = cms.EndPath(
    process.hcalTowerAnalyzer *
    process.hcalRecoAnalyzer *
    process.MEtoEDMConverter
    )

# Schedule definition
process.schedule = cms.Schedule(process.HLTSchedule)
process.schedule.extend([process.hcalHitsValidation_step,process.FEVTDEBUGHLToutput_step])

