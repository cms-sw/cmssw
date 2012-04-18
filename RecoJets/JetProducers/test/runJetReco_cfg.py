import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_4_2_0/RelValTTbar/GEN-SIM-RECO/MC_42_V9-v1/0054/3CBCB401-935E-E011-8345-0026189437F8.root'
#        '/store/mc/Fall10/QCD_Pt_15to3000_TuneZ2_Flat_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/16A40D7C-63CC-DF11-A8F9-E41F13181890.root'
#    '/store/relval/CMSSW_3_8_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_38Y_V9-v1/0018/4E074A51-97AF-DF11-AED9-003048678ED2.root',
#    '/store/relval/CMSSW_3_8_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_38Y_V9-v1/0018/9AC16F8D-A7AF-DF11-80AC-003048D15E14.root',
#    '/store/relval/CMSSW_3_8_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_38Y_V9-v1/0018/9C41071C-96AF-DF11-AF18-003048679228.root',
#    '/store/relval/CMSSW_3_8_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_38Y_V9-v1/0018/B494B961-B0AF-DF11-8508-003048678B5E.root',
#    '/store/relval/CMSSW_3_8_2/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_38Y_V9-v1/0018/D07EF07A-9BAF-DF11-A708-003048678BAE.root',
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
process.output.outputCommands =  cms.untracked.vstring('drop *_*_*_RECO')
#process.output.outputCommands.append('keep recoCaloJets_*_*_*')
#process.output.outputCommands.append('keep recoPFJets_*_*_*')
#process.output.outputCommands.append('keep recoGenJets_*_*_*')
#process.output.outputCommands.append('keep recoBasicJets_*_*_*')
process.output.outputCommands.append('keep *_*_*_JETRECO')
#process.output.outputCommands.append('keep *_trackRefsForJets_*_*')
#process.output.outputCommands.append('keep *_generalTracks_*_*')

# jet reconstruction
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START311_V1::All'

process.load("RecoJets/Configuration/RecoPFClusterJets_cff")

process.load("RecoJets/JetAssociationProducers/trackExtrapolator_cfi")

#process.recoJets = cms.Path(process.trackExtrapolator+process.jetGlobalReco+process.CastorFullReco+process.jetHighLevelReco+process.recoPFClusterJets)
process.recoJets = cms.Path(process.trackExtrapolator+process.jetGlobalReco+process.CastorFullReco+process.jetHighLevelReco+process.recoPFClusterJets)

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
