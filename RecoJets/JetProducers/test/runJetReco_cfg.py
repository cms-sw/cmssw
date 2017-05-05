import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
process.source = cms.Source("PoolSource", fileNames = filesRelValTTbarPileUpGENSIMRECO )
#fileNames = cms.untracked.vstring(
#'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/224918A9-C63F-E411-9E41-0025905A6118.root',
#'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/80D36272-BB3F-E411-A865-0026189437EB.root',
#'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/B449F5E7-BD3F-E411-A643-002618943874.root',
#'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/E8CC1FE0-CA3F-E411-9060-0025905B8592.root'
#    ))
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

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
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load("RecoJets/Configuration/RecoJetsGlobal_cff")

process.recoJets = cms.Path(#process.recoPFJetsWithSubstructure)
		process.fixedGridRhoAll+process.fixedGridRhoFastjetAll+process.fixedGridRhoFastjetCentral#+process.fixedGridRhoFastjetCentralChargedPileUp+process.fixedGridRhoFastjetCentralNeutral) #
+process.ak4PFJets) #+process.ak5PFJets+process.ak7PFJets+process.ak8PFJets+process.ca4PFJets+process.ca8PFJets+process.goodOfflinePrimaryVertices+process.pfPileUpJME+process.pfNoPileUpJME+process.ak5PFJetsCHS+process.ak5PFJetsCHSPruned+process.ak5PFJetsCHSFiltered+process.ak5PFJetsCHSTrimmed+process.ak5PFJetsCHSSoftDrop+process.ak4PFJetsCHS+process.ak8PFJetsCHS+process.ak8PFJetsCHSPruned+process.ak8PFJetsCHSFiltered+process.ak8PFJetsCHSTrimmed+process.ak8PFJetsCHSSoftDrop+process.ak8PFJetsCHSConstituents+process.ak8PFJetsCHSPrunedMass+process.ak8PFJetsCHSTrimmedMass+process.ak8PFJetsCHSSoftDropMass+process.ak8PFJetsCHSFilteredMass+process.ca8PFJetsCHS+process.ca8PFJetsCHSPruned+process.ca8PFJetsCHSFiltered+process.ca8PFJetsCHSTrimmed+process.ca8PFJetsCHSSoftDrop+process.cmsTopTagPFJetsCHS+process.hepTopTagPFJetsCHS+process.ca15PFJetsCHSMassDropFiltered+process.ca15PFJetsCHSFiltered+process.ak8PFJetsCS+process.ak8PFJetsCSConstituents+process.ak8PFJetsCSPruned+process.ak8PFJetsCSTrimmed+process.ak8PFJetsCSFiltered+process.puppi+process.ak4PFJetsPuppi+process.softKiller+process.ak4PFJetsSK)

process.out = cms.EndPath(process.output)

# schedule
process.schedule = cms.Schedule(process.recoJets,process.out)
