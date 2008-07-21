import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/20/RelVal-RelValZTT-1213920853/0000/2880E97B-893E-DD11-A98B-001617C3B79A.root', 
        '/store/relval/2008/6/20/RelVal-RelValZTT-1213920853/0000/32B08924-893E-DD11-ABD4-000423D94534.root', 
        '/store/relval/2008/6/20/RelVal-RelValZTT-1213920853/0000/38CB1BCE-863E-DD11-A69B-001617DBD332.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)
process.hltanalysis = cms.EDAnalyzer("HLTAnalyzer",
    mctruth = cms.string('genParticles'),
    muon = cms.string('muons'),
    calotowers = cms.string('towerMaker'),
    l1GtObjectMapRecord = cms.string('hltL1GtObjectMap'),
    Photon = cms.string('correctedPhotons'),
    Electron = cms.string('pixelMatchGsfElectrons'),
    ht = cms.string('htMet'),
    l1extramc = cms.string('hltL1extraParticles'),
    genmet = cms.string('genMet'),
    l1GtReadoutRecord = cms.string('hltGtDigis'),
    l1GctCounts = cms.InputTag("hltGctDigis"),
    recmet = cms.string('met'),
    hltresults = cms.string('TriggerResults'),
    recjets = cms.string('MCJetCorJetIcone5'),
    genEventScale = cms.string('genEventScale'),
    RunParameters = cms.PSet(
        GenJetMin = cms.double(10.0),
        Monte = cms.bool(True),
        CalJetMin = cms.double(10.0),
        HistogramFile = cms.string('TEST_HLTAnalyzer.root'),
        EtaMin = cms.double(-5.2),
        Debug = cms.bool(False),
        EtaMax = cms.double(5.2)
    ),
    genjets = cms.string('iterativeCone5GenJets')
)

process.p1 = cms.Path(process.hltanalysis)


