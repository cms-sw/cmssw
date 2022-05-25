import FWCore.ParameterSet.Config as cms

process = cms.Process("REPROD")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['mc']
from Configuration.AlCa.autoCond import autoCond 
process.GlobalTag.globaltag = cms.string( autoCond[ 'startup' ] )
#process.GlobalTag.globaltag = 'START50_V10::All'

#process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
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
process.source.fileNames = [ '/store/relval/CMSSW_5_0_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START50_V8-v2/0073/324BAB7B-C328-E111-B624-00261894389E.root',
                             '/store/relval/CMSSW_5_0_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START50_V8-v2/0073/72BA0554-C328-E111-B2A6-002618943972.root',
                             '/store/relval/CMSSW_5_0_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START50_V8-v2/0073/B44EAD5D-C328-E111-8057-0018F3D096BC.root',
                             '/store/relval/CMSSW_5_0_0/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START50_V8-v2/0073/EE7D4C4A-0529-E111-84F5-002618943900.root'
    ]
    
process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_NoTracking_cff")
process.display = cms.OutputModule("PoolOutputModule",
                                   process.DisplayEventContent,
    fileName = cms.untracked.string('display.root')
)

# modify reconstruction sequence
process.pfTrack.MuColl = cms.InputTag('muons')
process.particleFlowTmp.muons = cms.InputTag('muons')
process.particleFlow.FillMuonRefs = False
#process.particleFlowTmp.useHO = True


# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.particleFlowCluster+
                                   process.ecalClusters)


process.globalReReco = cms.Sequence(process.particleFlowTrackWithDisplacedVertex+
                                    process.gsfEcalDrivenElectronSequence
                                    )

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

#process.load("RecoParticleFlow.PFProducer.particleFlowCandidateChecker_cfi")
#process.particleFlowCandidateChecker.pfCandidatesReco = cms.InputTag("particleFlow","","REPROD")
#process.particleFlowCandidateChecker.pfCandidatesReReco = cms.InputTag("particleFlow","","REPROD2")
#process.particleFlowCandidateChecker.pfJetsReco = cms.InputTag("ak5PFJets","","REPROD")
#process.particleFlowCandidateChecker.pfJetsReReco = cms.InputTag("ak5PFJets","","REPROD2")

# The complete reprocessing
process.p = cms.Path(process.localReReco+
                     process.globalReReco+
                     process.pfReReco+
                     process.genReReco
                     #+process.particleFlowCandidateChecker
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

process.MessageLogger.cerr.FwkReport.reportEvery = 100


