import FWCore.ParameterSet.Config as cms

process = cms.Process("MYGAMMAJET")

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['startup']

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    fileNames = 
cms.untracked.vstring('/store/relval/CMSSW_7_3_0/RelValPhotonJets_Pt_10_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/522CE329-7B81-E411-B6C3-0025905A6110.root')
)

process.load("Calibration.HcalAlCaRecoProducers.alcagammajet_cfi")

process.GammaJetRecos = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('drop *', 
#        'keep *_GammaJetProd_*_*'),
    fileName = cms.untracked.string('gjet.root')
)

process.p = cms.Path(process.GammaJetProd)                   
process.e = cms.EndPath(process.GammaJetRecos)
