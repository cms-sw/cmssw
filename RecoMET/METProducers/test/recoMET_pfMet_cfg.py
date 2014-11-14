import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
process = cms.Process("TEST")

##____________________________________________________________________________||
process.load("FWCore.MessageLogger.MessageLogger_cfi")

##____________________________________________________________________________||
process.load("RecoMET/METProducers.PFMET_cfi")
process.load("RecoMET/METProducers.METSignificanceParams_cfi")
process.load("RecoMET/METProducers.METSignificanceObjects_cfi")

##____________________________________________________________________________||
#from RecoMET.METProducers.testInputFiles_cff import recoMETtestInputFiles
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring("/store/relval/CMSSW_7_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/CE7E8152-FE59-E411-91D5-0025905A60F2.root")
    )

##____________________________________________________________________________||
process.out = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('recoMET_pfMet.root'),
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
process.pfMetWithSignificance = process.pfMet.clone(
    process.METSignificanceParams,
    calculateSignificance = cms.bool(True),
    jets = cms.InputTag("ak4PFJets"),
    leptons = cms.VInputTag("selectedElectrons", "selectedMuons", "selectedPhotons")
    )

##____________________________________________________________________________||
process.p = cms.Path(
    process.selectionSequenceForMETSig *
    process.pfMetWithSignificance *
    process.pfMet
    )

process.e1 = cms.EndPath(
    process.out
    )

##____________________________________________________________________________||
