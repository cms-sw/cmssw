import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_5_0_0/RelValTTbar/GEN-SIM-RECO/START50_V8-v3/0074/182BA6EB-BD2A-E111-801C-003048678A88.root',
        '/store/relval/CMSSW_5_0_0/RelValTTbar/GEN-SIM-RECO/START50_V8-v3/0074/8E79DF51-B92A-E111-A9B4-00261894396E.root',
        '/store/relval/CMSSW_5_0_0/RelValTTbar/GEN-SIM-RECO/START50_V8-v3/0074/AE46DE36-C02A-E111-AF78-003048FF9AC6.root',
        '/store/relval/CMSSW_5_0_0/RelValTTbar/GEN-SIM-RECO/START50_V8-v3/0082/5893EB84-542B-E111-AFDF-002618943915.root',
        '/store/relval/CMSSW_5_0_0/RelValTTbar/GEN-SIM-RECO/START50_V8-v3/0082/8C111086-542B-E111-AA93-002618943983.root',
        '/store/relval/CMSSW_5_0_0/RelValTTbar/GEN-SIM-RECO/START50_V8-v3/0082/E0196E85-542B-E111-AA08-0030486791DC.root'
    )
    #inputCommands = cms.untracked.vstring('keep *_*_*_*','drop recoTrackExtrapolations_*_*_RECO')  
    )
# output
process.load('Configuration/EventContent/EventContent_cff')
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = process.RecoJetsAOD.outputCommands, 
    fileName = cms.untracked.string('testJetRecoRECO.root'),
    dataset = cms.untracked.PSet(
            dataTier = cms.untracked.string(''),
                    filterName = cms.untracked.string('')
                    )
)
process.output.outputCommands.append('drop *_*_*_RECO')
#process.output.outputCommands.append('keep recoCaloJets_*_*_*')
#process.output.outputCommands.append('keep recoPFJets_*_*_*')
#process.output.outputCommands.append('keep recoGenJets_*_*_*')
#process.output.outputCommands.append('keep recoBasicJets_*_*_*')
process.output.outputCommands.append('keep *_*_*_JETRECO')
process.output.outputCommands.append('keep recoPFCandidates_particleFlow_*_*')
#process.output.outputCommands.append('keep *_trackRefsForJets_*_*')
#process.output.outputCommands.append('keep *_generalTracks_*_*')

# jet reconstruction
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START50_V10::All'

process.load("RecoJets/Configuration/RecoPFClusterJets_cff")
process.load("RecoMET/METProducers/PFClusterMET_cfi")

process.load("RecoJets/JetAssociationProducers/trackExtrapolator_cfi")

process.ca12PFJetsMassDropFiltered = process.ak5PFJetsMassDropFiltered.clone(
    rParam = cms.double(1.2)
    )

process.ak5PFJetsTrimmed.doAreaFastjet = True
process.ak5PFJetsPruned.doAreaFastjet = True
process.ca8PFJetsPruned = process.ak5PFJetsPruned.clone(
    jetAlgorithm = cms.string("CambridgeAachen"),
    rParam       = cms.double(0.8)
    )

#process.recoJets = cms.Path(process.trackExtrapolator+process.jetGlobalReco+process.CastorFullReco+process.jetHighLevelReco+process.recoPFClusterJets)
process.recoJets = cms.Path(process.ak5PFJets+
                            process.ak5PFJetsTrimmed+
                            process.ak5PFJetsFiltered+
                            process.ak5PFJetsMassDropFiltered+
                            process.ca12PFJetsMassDropFiltered+
                            process.ak5PFJetsPruned+
                            process.ca8PFJetsPruned
                            )





process.out = cms.EndPath(process.output)

# Set the threshold for output logging to 'info'
#process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
             limit = cms.untracked.int32(0)
)

# schedule
process.schedule = cms.Schedule(process.recoJets,process.out)
