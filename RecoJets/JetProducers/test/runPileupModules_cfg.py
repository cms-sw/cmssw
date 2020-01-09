import FWCore.ParameterSet.Config as cms

process = cms.Process("PURECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/224918A9-C63F-E411-9E41-0025905A6118.root',
'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/80D36272-BB3F-E411-A865-0026189437EB.root',
'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/B449F5E7-BD3F-E411-A643-002618943874.root',
'/store/relval/CMSSW_7_2_0_pre6/RelValRSKKGluon_m3000GeV_13/GEN-SIM-RECO/PRE_LS172_V11-v1/00000/E8CC1FE0-CA3F-E411-9060-0025905B8592.root'
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
process.output.outputCommands.append('keep *_*_*_PURECO')
process.output.outputCommands.append('keep recoPFCandidates_particleFlow_*_*')
#process.output.outputCommands.append('keep *_trackRefsForJets_*_*')
#process.output.outputCommands.append('keep *_generalTracks_*_*')

# jet reconstruction
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load('CommonTools.ParticleFlow.pfNoPileUpJME_cff')
process.load("RecoJets/JetProducers/ak8PFJetsCS_cfi")
process.load("RecoJets/JetProducers/ak8PFJetsCHSCS_cfi")
process.load('CommonTools.PileupAlgos.softKiller_cfi')

process.recoPU = cms.Path(process.pfNoPileUpJMESequence*process.ak8PFJetsCS*process.ak8PFJetsCHSCS*process.particleFlowSKPtrs)

process.out = cms.EndPath(process.output)

# schedule
process.schedule = cms.Schedule(process.recoPU,process.out)
