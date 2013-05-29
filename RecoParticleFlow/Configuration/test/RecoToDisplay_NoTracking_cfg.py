import FWCore.ParameterSet.Config as cms

process = cms.Process("REPROD")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['mc']
from Configuration.AlCa.autoCond import autoCond 
process.GlobalTag.globaltag = cms.string( autoCond[ 'startup' ] )



#process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(),
    eventsToProcess = cms.untracked.VEventRange(),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

from RecoParticleFlow.Configuration.reco_QCDForPF_cff import fileNames
process.source.fileNames = ['/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0000/0EE664B2-FFA3-E011-918F-002618943882.root',
                            '/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0000/644190E4-F0A3-E011-BED0-00304867C1BA.root',
                            '/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0000/9A80E178-18A4-E011-B1DD-002618FDA287.root',
                            '/store/relval/CMSSW_4_4_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START43_V4-v1/0005/F29B164E-43A6-E011-B2B1-00248C55CC7F.root']
    

process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_NoTracking_cff")
process.display = cms.OutputModule("PoolOutputModule",
                                   process.DisplayEventContent,
    fileName = cms.untracked.string('display.root')
)

# modify reconstruction sequence
#process.hbhereflag = process.hbhereco.clone()
#process.hbhereflag.hbheInput = 'hbhereco'
#process.towerMakerPF.hbheInput = 'hbhereflag'
#process.particleFlowRecHitHCAL.hcalRecHitsHBHE = cms.InputTag("hbhereflag")
process.particleFlowTmp.muons = cms.InputTag('muons')
process.particleFlow.FillMuonRefs = False 

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.particleFlowCluster)


# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco+
                                process.particleFlowLinks+
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

#process.pfConversions.conversionCollection = cms.InputTag("trackerOnlyConversions", "")

# The complete reprocessing
process.p = cms.Path(process.localReReco+
                     process.particleFlowTrackWithDisplacedVertex+
                     process.gsfEcalDrivenElectronSequence+
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

process.MessageLogger.cerr.FwkReport.reportEvery = 1000


