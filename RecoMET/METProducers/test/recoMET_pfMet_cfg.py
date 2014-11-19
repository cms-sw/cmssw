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
    fileNames = cms.untracked.vstring(
       #"/store/relval/CMSSW_7_3_0_pre1/RelValZpMM_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/A8782A3D-0A5A-E411-A5FF-0025905AA9F0.root",
       #"/store/relval/CMSSW_7_3_0_pre1/RelValZpMM_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/BA64CF78-085A-E411-B4A1-0025905964C2.root"
       #"/store/relval/CMSSW_7_3_0_pre1/RelValMinBias_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/22256348-FF59-E411-A2AE-0025905A6090.root",
       #"/store/relval/CMSSW_7_3_0_pre1/RelValMinBias_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/3C4078FF-F359-E411-BB95-0025905A48D8.root",
       #"/store/relval/CMSSW_7_3_0_pre1/RelValMinBias_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/5097130F-FC59-E411-9613-0025905A60D2.root"
       #"/store/relval/CMSSW_7_3_0_pre1/RelValZMM_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/3AC25AA2-075A-E411-A24B-002618FDA265.root",
       #"/store/relval/CMSSW_7_3_0_pre1/RelValZMM_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/98214EA1-075A-E411-9F19-0025905A609A.root"
       #"root://xrootd.unl.edu//store/data/Run2012A/DoubleMu/AOD/22Jan2013-v1/20000/001AE30A-BA81-E211-BBE7-003048FFD770.root"
       "/store/relval/CMSSW_7_3_0_pre1/RelValZEE_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/C2212C6F-EB59-E411-AB78-0025905B85EE.root",
       "/store/relval/CMSSW_7_3_0_pre1/RelValZEE_13/GEN-SIM-RECO/PRE_LS172_V15-v1/00000/F0DAE66D-EB59-E411-95AF-0025905A6088.root"
       )
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
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

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
