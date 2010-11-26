import FWCore.ParameterSet.Config as cms
#import os

process = cms.Process("REPROD")

# General
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

# Global tag for 39X (UPDATE FOR LATER CMSSW VERSIONS)
process.GlobalTag.globaltag = 'GR_R_39X_V1::All'

# Other statements for 39X (UPDATE FOR LATER CMSSW VERSIONS)
from Configuration.GlobalRuns.reco_TLR_39X import customisePPData
customisePPData(process)

## particle flow HF cleaning
#process.particleFlowRecHitHCAL.LongShortFibre_Cut = 30.
#process.particleFlowRecHitHCAL.ApplyTimeDPG = False
#process.particleFlowRecHitHCAL.ApplyPulseDPG = True
#process.particleFlowRecHitECAL.timing_Cleaning = True

# Event file to process
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
      'file:/tmp/pjanot/jordan.root',
      ),
    #eventsToProcess = cms.untracked.VEventRange('143827:62146418-143827:62146418'),
    )
process.source.secondaryFileNames = cms.untracked.vstring()
process.source.noEventSort = cms.untracked.bool(True)
process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

# Number of events to process
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# This is for filtering on L1 technical trigger bit: MB and no beam halo
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('(0 AND (36 OR 37 OR 38 OR 39))')

process.scrapping = cms.EDFilter("FilterOutScraping",
                                applyfilter = cms.untracked.bool(True),
                                debugOn = cms.untracked.bool(False),
                                numtrack = cms.untracked.uint32(10),
                                thresh = cms.untracked.double(0.25)
                                )

process.load('CommonTools/RecoAlgos/HBHENoiseFilter_cfi')

process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    fileName = cms.untracked.string('jordan.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
)

process.load("Configuration.EventContent.EventContent_cff")
process.rereco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('reco.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
)

# Maxime !!!@$#^%$^%#@
#process.particleFlowDisplacedVertexCandidate.verbose = False
#process.particleFlowDisplacedVertex.verbose = False

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
#process.towerMakerPF.HcalAcceptSeverityLevel = 11

# Tests for John-Paul Chou hbhe cleaning
#process.hbhereflag = process.hbhereco.clone()
#process.hbhereflag.hbheInput = 'hbhereco'
#process.towerMakerPF.hbheInput = 'hbhereflag'
#process.particleFlowRecHitHCAL.hcalRecHitsHBHE = cms.InputTag("hbhereflag")

process.localReReco = cms.Sequence(process.siPixelRecHits+
                                   process.siStripMatchedRecHits+
                                   #process.hbhereflag+
                                   process.particleFlowCluster)

#Photon re-reco
process.photonReReco = cms.Sequence(process.conversionSequence+
                                    process.trackerOnlyConversionSequence+
                                    process.photonSequence+
                                    process.photonIDSequence)

# Track re-reco
process.globalReReco =  cms.Sequence(process.offlineBeamSpot+
                                     process.recopixelvertexing+
                                     process.ckftracks+
                                     process.ctfTracksPixelLess+
                                     process.offlinePrimaryVertices *
                                     process.offlinePrimaryVerticesWithBS *
                                     process.caloTowersRec+
                                     process.vertexreco+
                                     process.recoJets+
                                     process.muonrecoComplete+
                                     process.electronGsfTracking+
                                     process.photonReReco+
                                     process.metreco)



# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco+
                                process.recoPFJets+
                                process.recoPFMET+
                                process.PFTau#+
#                                process.particleFlowDisplacedVertexCandidate+
#                                process.particleFlowDisplacedVertex
                                )
                                
# Gen Info re-processing
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.Configuration.RecoGenJets_cff")
process.load("RecoMET.Configuration.GenMETParticles_cff")
process.load("RecoMET.Configuration.RecoGenMET_cff")
process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")
process.load("RecoParticleFlow.Configuration.HepMCCopy_cfi")
process.genReReco = cms.Sequence(process.generator+
                                 process.genParticles+
                                 process.genJetParticles+
                                 process.recoGenJets+
                                 process.genMETParticles+
                                 process.recoGenMET+
                                 process.particleFlowSimParticle)

# The complete reprocessing
process.p = cms.Path(process.scrapping+
                     process.HBHENoiseFilter+
                     process.localReReco+
                     process.globalReReco+
                     process.pfReReco
                     )

# And the output.
# Write out only filtered events
process.display.SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') )
process.rereco.SelectEvents = cms.untracked.PSet( SelectEvents = cms.vstring('p') )
#process.outpath = cms.EndPath(process.rereco+process.display)
process.outpath = cms.EndPath(process.display)


# Schedule the paths
process.schedule = cms.Schedule(
    process.p,
    process.outpath
)

# And the logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    #fileMode = cms.untracked.string('NOMERGE'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1

