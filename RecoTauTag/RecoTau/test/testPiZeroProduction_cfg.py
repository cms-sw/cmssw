import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.source = cms.Source ("PoolSource", fileNames = cms.untracked.vstring(
    #'/store/relval/CMSSW_3_8_1/RelValZTT/GEN-SIM-RECO/START38_V8-v1/0011/F4817DF9-E9A1-DF11-AA95-0026189437E8.root',
    #'/store/relval/CMSSW_3_8_1/RelValZTT/GEN-SIM-RECO/START38_V8-v1/0011/DA56125E-EAA1-DF11-9E00-001A92971B9A.root',
    '/store/relval/CMSSW_3_8_1/RelValZTT/GEN-SIM-RECO/START38_V8-v1/0011/3AD9A89B-E8A1-DF11-9841-0018F3D096C6.root',
    '/store/relval/CMSSW_3_8_1/RelValZTT/GEN-SIM-RECO/START38_V8-v1/0011/3AD57993-28A2-DF11-9856-0018F3D09704.root',
    '/store/relval/CMSSW_3_8_1/RelValZTT/GEN-SIM-RECO/START38_V8-v1/0011/1C21E56A-EAA1-DF11-8CDA-0018F3D09624.root'
))

process.load("RecoTauTag.RecoTau.RecoTauPiZeroProducer_cfi")

process.load("RecoTauTag.RecoTau.RecoTauCombinatoricProducer_cfi")
process.load("RecoTauTag.RecoTau.RecoTauHPSTancTauProdcuer_cfi")
process.load("RecoTauTag.RecoTau.RecoTauShrinkingConeProducer_cfi")
process.load("RecoTauTag.TauTagTools.TauTruthProduction_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "MC_36Y_V10::All"

process.dumpTruth = cms.EDAnalyzer(
    "TauGenJetDumper",
    src = cms.InputTag("trueHadronicTaus")
)

process.dumpReco = cms.EDAnalyzer(
    "RecoTauDumper",
    src = cms.InputTag("shrinkingConeRecoTaus")
)

# Dump all the RecoTauPiZeros associated to the input jets into a plain vector
process.flattenPiZeros = cms.EDProducer(
    "RecoTauPiZeroFlattener",
    jetSrc = cms.InputTag("ak5PFJets"),
    piZeroSrc = cms.InputTag("ak5PFJetsRecoTauPiZeros"),
)

process.dumpContent = cms.EDAnalyzer('EventContentAnalyzer')


process.path = cms.Path(
     process.ak5PFJetsRecoTauPiZeros
    +process.flattenPiZeros
    +process.dumpContent
    #+process.combinatoricRecoTaus
    #+process.shrinkingConeRecoTaus
    #+process.tauTruthSequence
    #+process.dumpTruth
    #+process.dumpReco
    #+process.hpsTancRecoTausSequence
)
