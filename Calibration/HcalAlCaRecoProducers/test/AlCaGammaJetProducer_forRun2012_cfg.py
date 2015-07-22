import FWCore.ParameterSet.Config as cms

process = cms.Process("MYGAMMAJET")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run1_data']

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery=cms.untracked.int32(1000)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(20)
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring(
#   'file:/tmp/andriusj/6EC8FCC8-E2A8-E411-9506-002590596468.root'
#        '/store/relval/CMSSW_7_4_0_pre6/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/6EC8FCC8-E2A8-E411-9506-002590596468.root'
    'file:/tmp/andriusj/Run2012A_Photon_22Jan2013-002618943913.root'
 )
)

process.load("Calibration.HcalAlCaRecoProducers.alcagammajet_cfi")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_Output_cff")

process.GammaJetProd.PhoInput= cms.InputTag("photons")
process.GammaJetProd.gsfeleInput= cms.InputTag("gsfElectrons")
process.GammaJetProd.PhoLoose= cms.InputTag("PhotonIDProd","PhotonCutBasedIDLoose")
process.GammaJetProd.PhoTight= cms.InputTag("PhotonIDProd","PhotonCutBasedIDTight")

#process.GammaJetRecos = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('drop *',
##                 'keep recoPhotonCores_*_*_*',
#                 'keep recoSuperClusters_*_*_*',
#                 #'keep recoTracks_*_*_*',
#                 'keep recoTracks_generalTracks_*_*',
#                 #'keep *_PhotonIDProd_*_*',
#               'keep *_particleFlow_*_*',
#              'keep recoPFBlocks_particleFlowBlock_*_*',
#              'keep recoPFClusters_*_*_*',
##                         'keep *_particleFlowPtrs_*_*',
#        'keep *_GammaJetProd_*_*'),
#    fileName = cms.untracked.string('gjet.root')
#)

process.GammaJetRecos = cms.OutputModule("PoolOutputModule",
   outputCommands = process.OutALCARECOHcalCalGammaJet.outputCommands,
   fileName = cms.untracked.string('/tmp/andriusj/gjet_Run2012A.root')
)

##process.GammaJetRecos.outputCommands.append('keep *_particleFlowPtrs_*_*')

process.load('RecoJets.Configuration.RecoPFJets_cff')
process.load('CommonTools.ParticleFlow.PF2PAT_cff')
process.load("CommonTools.ParticleFlow.pfNoPileUp_cff")

process.seq_ak4PFCHS= cms.Sequence( process.particleFlowPtrs *
                                    process.pfNoPileUpJMESequence *
                                    process.ak4PFJetsCHS )

process.p = cms.Path( process.seq_ak4PFCHS * process.GammaJetProd )
process.e = cms.EndPath(process.GammaJetRecos)
