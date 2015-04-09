import FWCore.ParameterSet.Config as cms

process = cms.Process("MYGAMMAJET")

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond_condDBv2 import autoCond
process.GlobalTag.globaltag=autoCond['run2_mc']

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(100)
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring(
#   'file:/tmp/andriusj/6EC8FCC8-E2A8-E411-9506-002590596468.root'
        '/store/relval/CMSSW_7_4_0_pre6/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/6EC8FCC8-E2A8-E411-9506-002590596468.root'
 )
)

process.load("Calibration.HcalAlCaRecoProducers.alcagammajet_cfi")
process.load("Calibration.HcalAlCaRecoProducers.ALCARECOHcalCalGammaJet_Output_cff")

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
   fileName = cms.untracked.string('gjet.root')
)

process.p = cms.Path(process.GammaJetProd)
process.e = cms.EndPath(process.GammaJetRecos)
