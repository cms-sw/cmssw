import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("TEST")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")

##____________________________________________________________________________||
process.load("RecoMET.METProducers.CaloMET_cfi")
process.load("RecoMET.METProducers.METSigParams_cfi")
process.load("RecoMET.METProducers.caloMetM_cfi")
process.load("RecoJets.Configuration.CaloTowersRec_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("RecoLocalCalo.Configuration.RecoLocalCalo_cff")

##____________________________________________________________________________||
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

##____________________________________________________________________________||
from RecoMET.METProducers.testInputFiles_cff import recoMETtestInputFiles

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(recoMETtestInputFiles)
    )

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('recoMET_caloMet.root'),
    SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_*_*_TEST'
        )
    )

##____________________________________________________________________________||
process.options   = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
process.MessageLogger.cerr.FwkReport.reportEvery = 50
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))

##____________________________________________________________________________||
process.caloMetWithSignificance = process.caloMet.clone(
    process.METSignificance_params,
    calculateSignificance = cms.bool(True)
    )

##____________________________________________________________________________||
process.p = cms.Path(
    process.towerMakerWithHO *
    process.caloMet *
    process.caloMetBEFO *
    process.caloMetBE *
    process.caloMetBEO *
    process.caloMetM *
    process.caloMetWithSignificance
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
