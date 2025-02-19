import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")
process.load("RecoJets.Configuration.CaloTowersRec_cff")

process.load("RecoJets.Configuration.RecoGenJets_cff")

process.load("RecoJets.JetProducers.GenJetParameters_cfi")

process.load("RecoJets.JetProducers.CaloJetParameters_cfi")

process.load("RecoJets.JetProducers.FastjetParameters_cfi")

process.load("RecoJets.JetProducers.ExtKtJetParameters_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(8000000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/056/591/F2F1483E-416F-DD11-A270-001617E30CE8.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/313/4214986A-196D-DD11-BED7-000423D992DC.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/381/0A45CF4D-336D-DD11-A6EA-000423D6CAF2.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/381/FE634B34-316D-DD11-BABF-000423D6B358.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/394/C6CC14EE-316D-DD11-A3E8-000423D986A8.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/400/84750050-416D-DD11-9230-000423D6B48C.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/144C1DEA-466D-DD11-9A1F-000423D9880C.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/16ED5E7C-486D-DD11-A428-000423D6C8EE.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/2403B5E5-466D-DD11-B0F3-001617DBD332.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/3A9E9CB3-496D-DD11-A88D-001617E30CE8.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/3CCFE9B5-496D-DD11-B645-001617C3B778.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/564598B3-496D-DD11-8403-001617C3B76E.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/6639F8FF-486D-DD11-B710-000423D6CA6E.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/6E27F1B5-496D-DD11-8265-001617C3B5F4.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET4_v1/000/057/404/76CF2D2C-4B6D-DD11-B626-000423D98DB4.root'
    )
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoCaloJets_*_*_*', 
        'keep recoGenJets_*_*_*'),
    fileName = cms.untracked.string('sisConeJets.root')
)

process.myOut = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('f')
    ),
    fileName = cms.untracked.string('FilteredEvents.root')
)

process.TFileService = cms.Service("TFileService",
    closeFileFast = cms.untracked.bool(True),
    fileName = cms.string('myhisto.root')
)

process.compare = cms.EDFilter("myJetAna",
    GenJetAlgorithm     = cms.string('sisCone5GenJets'),
    CaloJetAlgorithm    = cms.string('sisCone5CaloJets'),
    TriggerResultsLabel = cms.InputTag("TriggerResults::HLT")                               
)

process.filter = cms.EDFilter("myFilter",
    CaloJetAlgorithm = cms.string('sisCone5CaloJets'),
    TriggerResultsLabel = cms.InputTag("TriggerResults::HLT")                               
)

process.evtInfo = cms.OutputModule("AsciiOutputModule")

process.f = cms.Path(process.filter)
process.p = cms.Path(process.compare)
process.e = cms.EndPath(process.myOut*process.evtInfo)


