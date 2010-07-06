import FWCore.ParameterSet.Config as cms

process = cms.Process("REPROD")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']
#process.GlobalTag.globaltag = 'MC_3XY_V25::All'
#process.load("Configuration.StandardSequences.FakeConditions_cff")

#process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
      #'rfio:/castor/cern.ch/user/r/rebeca/Feb10PFMET/hww2l_RECOSIM.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_0.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_1.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_2.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_3.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_4.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_5.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_6.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_7.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_8.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_9.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_10.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_11.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_12.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_13.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_14.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_15.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_16.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_17.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_18.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_19.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_20.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_21.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_22.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_23.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_24.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_25.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_26.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_27.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_28.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_29.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_30.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_31.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_32.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_33.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_34.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_35.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_36.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_37.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_38.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_39.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_40.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_41.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_42.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_43.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_44.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_45.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_46.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_47.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_48.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_49.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_50.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_51.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_52.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_53.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_54.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_55.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_56.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_57.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_58.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_59.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_60.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_61.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_62.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_63.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_64.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_65.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_66.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_67.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_68.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_69.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_70.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_71.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_72.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_73.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_74.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_75.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_76.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_77.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_78.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_79.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_80.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_81.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_82.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_83.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_84.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_85.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_86.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_87.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_88.root',
      'rfio:/castor/cern.ch/user/p/pjanot/CMSSW360pre6/reco_QCDForPF_Full_89.root',
      ),
    #eventsToProcess = cms.untracked.VEventRange('1:195-1:200'),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

#process.MessageLogger = cms.Service("MessageLogger",
#    rectoblk = cms.untracked.PSet(
#        threshold = cms.untracked.string('INFO')
#    ),
#    destinations = cms.untracked.vstring('rectoblk')
#)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    fileName = cms.untracked.string('display.root')
)

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.siPixelRecHits+
                                   process.siStripMatchedRecHits+
                                   process.particleFlowCluster)

# Track re-reco
process.globalReReco =  cms.Sequence(process.offlineBeamSpot+
                                     process.recopixelvertexing+
                                     process.ckftracks+
                                     process.caloTowersRec+
                                     process.vertexreco+
                                     process.recoJets+
                                     process.muonrecoComplete+
                                     process.electronGsfTracking+
                                     process.metreco)

# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco+
                                process.recoPFJets+
                                process.recoPFMET+
                                process.PFTau)
                                
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
process.p = cms.Path(process.localReReco+
                     process.globalReReco+
                     process.pfReReco+
                     process.genReReco
                     )

# And the output.
process.outpath = cms.EndPath(process.display)

# And the logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
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


