import FWCore.ParameterSet.Config as cms
import sys

infile_name = sys.argv[1]
outfile_name = '/afs/cern.ch/user/l/lgray/work/public/CMSSW_7_0_0_pre3_singlegconv/src/RecoParticleFlow/Configuration/test/%s/superClusterDump_%i.root'%(sys.argv[4],int(sys.argv[2]))
nevents = int(sys.argv[3])


process = cms.Process("REPROD")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['phase1_2022_realistic']

#process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(nevents)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    'root://cms-xrd-global.cern.ch/%s'%infile_name
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


# modify reconstruction sequence
#process.hbhereflag = process.hbhereco.clone()
#process.hbhereflag.hbheInput = 'hbhereco'
#process.towerMakerPF.hbheInput = 'hbhereflag'
#process.particleFlowRecHitHCAL.hcalRecHitsHBHE = cms.InputTag("hbhereflag")
process.particleFlowTmp.useHO = False

process.TFileService = cms.Service(
    "TFileService",
    fileName=cms.string(outfile_name)
    )

process.pfElectronSCTree = cms.EDAnalyzer(
    "PFSuperClusterTreeMaker",
    doGen = cms.untracked.bool(True),
    genSrc = cms.InputTag("genParticles"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    superClusterSrcEB = cms.InputTag('pfElectronTranslator','pf')
    )

process.pfPhotonSCTree = cms.EDAnalyzer(
    "PFSuperClusterTreeMaker",
    doGen = cms.untracked.bool(True),
    genSrc = cms.InputTag("genParticles"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    superClusterSrcEB = cms.InputTag('pfPhotonTranslator','pfphot')
    )

process.pfMustacheSCTree = cms.EDAnalyzer(
    "PFSuperClusterTreeMaker",
    doGen = cms.untracked.bool(True),
    genSrc = cms.InputTag("genParticles"),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    superClusterSrcEB = cms.InputTag("particleFlowSuperClusterECAL",
                                     "particleFlowSuperClusterECALBarrel"),
    superClusterSrcEE = cms.InputTag("particleFlowSuperClusterECAL",
                                     "particleFlowSuperClusterECALEndcapWithPreshower")
    )

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.siPixelRecHits+
                                   process.siStripMatchedRecHits+
                                   #process.hbhereflag+
                                   process.particleFlowCluster+
                                   process.ecalClusters)

# Track re-reco
process.globalReReco =  cms.Sequence(process.offlineBeamSpot+
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
                                process.PFTau+
                                process.pfElectronSCTree+
                                process.pfPhotonSCTree+
                                process.pfMustacheSCTree)

                                
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

# pf reprocessing
process.p = cms.Path(process.localReReco+
                     process.globalReReco+
                     process.pfReReco+
                     process.genReReco
                     #+process.particleFlowCandidateChecker
                     )

# And the logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.cout = cms.untracked.PSet(
#    threshold = cms.untracked.string('INFO')
#    )
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(True),
    #wantSummary = cms.untracked.bool(True),
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


