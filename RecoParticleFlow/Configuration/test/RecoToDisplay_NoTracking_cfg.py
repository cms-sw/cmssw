import FWCore.ParameterSet.Config as cms

process = cms.Process("REPROD")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#from Configuration.PyReleaseValidation.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['startup']

# Global tag for 39X (UPDATE FOR LATER CMSSW VERSIONS)
process.GlobalTag.globaltag = 'START311_V1A::All'

#process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/F840598D-CC36-E011-A0AB-003048678FB8.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/D4ADBCB6-BF36-E011-ADA3-00261894389D.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/BEE4681C-2D37-E011-A6F0-00304867BF18.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/9ED11227-D936-E011-A931-0018F3D0970C.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/882604A9-CC36-E011-9D3D-003048678D6C.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/841B578D-CC36-E011-9DF7-003048678E52.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/7A291736-D836-E011-BE53-0018F3D0962E.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/784149AD-D736-E011-8B0B-0018F3D0965C.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/584A23E6-6237-E011-85A3-0026189438F7.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/583EBB4C-D536-E011-BCF8-002354EF3BCE.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/52DB0C71-D236-E011-BBFC-00304867903E.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/2C3C9B8F-CC36-E011-B0BE-001A92971B32.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/14FF49B0-D936-E011-8CDB-0018F3D0970C.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/0444CBBB-D636-E011-8628-003048678F62.root',
    '/store/relval/CMSSW_3_11_0/RelValFlatQCD_Pt15_3000/GEN-SIM-RECO/START311_V1A-v1/0000/02D9E469-CC36-E011-99C4-003048678F0C.root'
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


process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_NoTracking_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    #outputCommands = cms.untracked.vstring('keep *'),
    #process.RECOSIMEventContent,
    fileName = cms.untracked.string('display_All.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
)

# modify reconstruction sequence
#process.hbhereflag = process.hbhereco.clone()
#process.hbhereflag.hbheInput = 'hbhereco'
#process.towerMakerPF.hbheInput = 'hbhereflag'
#process.particleFlowRecHitHCAL.hcalRecHitsHBHE = cms.InputTag("hbhereflag")

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.particleFlowCluster)


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
                     process.pfReReco+
                     process.genReReco
                     #+process.pfChargedHadronAnalyzer
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


