import FWCore.ParameterSet.Config as cms
import sys

process = cms.Process("REPROD")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']
#process.GlobalTag.globaltag = 'START50_V10::All'

#process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(    
    #'root://eoscms//eos/cms/store/relval/CMSSW_5_2_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START52_V1-v1/0105/2AAA5F86-8D57-E111-B6E8-003048678B84.root',
    #'root://eoscms//eos/cms/store/relval/CMSSW_5_2_0_pre5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/START52_V1-v1/0105/38D32839-8A57-E111-849D-0026189438E4.root'
    #'root://cms-xrd-global.cern.ch//store/relval/CMSSW_6_1_0/SingleGammaPt300ExtRelVal610/GEN-SIM-RECO/START61_V8_NoPuCustomEvC-v1/00000/00110DD9-9390-E211-88AB-5404A63886E6.root'
    '/store/relval/CMSSW_7_0_0_pre5/RelValSingleGammaPt35/GEN-SIM-RECO/PRE_ST62_V8-v1/00000/00FB34CE-D32F-E311-8968-002354EF3BDE.root'
    ),
    eventsToProcess = cms.untracked.VEventRange(),
    #eventsToProcess = cms.untracked.VEventRange('1:1217421-1:1217421'),
    #                                             '1:1220344-1:1220344',
    #                                             '1:1655912-1:1655912',
    #                                             '1:415027-1:415027',
    #                                             '1:460640-1:460640',
    #                                             '1:2054772-1:2054772'),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

#from RecoParticleFlow.Configuration.reco_QCDForPF_cff import fileNames
#process.source.fileNames = fileNames

process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    #outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('display.root')
)

process.load("Configuration.EventContent.EventContent_cff")
process.reco = cms.OutputModule("PoolOutputModule",
    process.RECOSIMEventContent,
    fileName = cms.untracked.string('/afs/cern.ch/user/l/lgray/work/public/phogun_35GeV.root')
)

# modify reconstruction sequence
#process.hbhereflag = process.hbhereco.clone()
#process.hbhereflag.hbheInput = 'hbhereco'
#process.towerMakerPF.hbheInput = 'hbhereflag'
#process.particleFlowRecHitHCAL.hcalRecHitsHBHE = cms.InputTag("hbhereflag")
process.particleFlowTmp.useHO = False

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.siPixelRecHits+
                                   process.siStripMatchedRecHits+
                                   #process.hbhereflag+
                                   process.particleFlowCluster+
                                   process.ecalClusters)

# Track re-reco
process.globalReReco =  cms.Sequence(process.offlineBeamSpot+
                                     process.MeasurementTrackerEvent+
                                     process.recopixelvertexing+
                                     process.ckftracks+
                                     process.caloTowersRec+
                                     process.vertexreco+
                                     process.recoJets+
                                     process.muonrecoComplete+
                                     process.muoncosmicreco+
                                     process.egammaGlobalReco+
                                     process.pfTrackingGlobalReco+
                                     process.egammaHighLevelRecoPrePF+
                                     process.muoncosmichighlevelreco+
                                     process.metreco)

# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco+
                                process.egammaHighLevelRecoPostPF+
                                process.muonshighlevelreco+
                                process.particleFlowLinks+
                                process.recoPFJets+
                                process.recoPFMET+
                                process.PFTau)
#pfeg switch
def switch_on_pfeg(the_process):
    the_process.particleFlowTmp.useEGammaFilters = cms.bool(True)
    the_process.particleFlowTmp.usePFPhotons = cms.bool(False)
    the_process.particleFlowTmp.usePFElectrons = cms.bool(False)
    the_process.particleFlow.GsfElectrons = cms.InputTag('gedGsfElectrons')
    the_process.particleFlow.Photons = cms.InputTag('gedPhotons')
    the_process.particleFlowReco.remove(the_process.pfElectronTranslatorSequence)
    the_process.particleFlowReco.remove(the_process.pfPhotonTranslatorSequence)
    the_process.egammaHighLevelRecoPostPF.remove(the_process.gsfElectronMergingSequence)
    the_process.interestingEleIsoDetIdEB.emObjectLabel = \
                                              cms.InputTag('gedGsfElectrons')
    the_process.interestingEleIsoDetIdEE.emObjectLabel = \
                                              cms.InputTag('gedGsfElectrons')
    the_process.interestingGamIsoDetIdEB.emObjectLabel = \
                                              cms.InputTag('gedPhotons')
    the_process.interestingGamIsoDetIdEE.emObjectLabel = \
                                              cms.InputTag('gedPhotons')
    the_process.PhotonIDProd.photonProducer = cms.string('gedPhotons')
    the_process.eidRobustLoose.src = cms.InputTag('gedGsfElectrons')
    the_process.eidRobustTight.src = cms.InputTag('gedGsfElectrons')
    the_process.eidRobustHighEnergy.src = cms.InputTag('gedGsfElectrons')
    the_process.eidLoose.src = cms.InputTag('gedGsfElectrons')
    the_process.eidTight.src = cms.InputTag('gedGsfElectrons')
switch_on_pfeg(process)

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
#process.particleFlowCandidateChecker.pfJetsReco = cms.InputTag("ak4PFJets","","REPROD")
#process.particleFlowCandidateChecker.pfJetsReReco = cms.InputTag("ak4PFJets","","REPROD2")
# The complete reprocessing
process.p = cms.Path(process.localReReco+
                     process.globalReReco+
                     process.pfReReco+
                     process.genReReco
                     #+process.particleFlowEGammaCandidateChecker
                     )

# And the output.
process.outpath = cms.EndPath(
    process.reco + 
    process.display
)

# And the monitoring
process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                        ignoreTotal=cms.untracked.int32(1),
                                        jobReportOutputOnly = cms.untracked.bool(True)
                                        )
process.Timing = cms.Service("Timing",
                             summaryOnly = cms.untracked.bool(True)
                             )

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
#process.MessageLogger.cout = cms.untracked.PSet(
#    threshold = cms.untracked.string('INFO')
#    )


