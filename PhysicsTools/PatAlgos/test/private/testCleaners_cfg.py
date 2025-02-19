# The following comments couldn't be translated into the new config version:

# discard jets that match with clean electrons
import FWCore.ParameterSet.Config as cms

process = cms.Process("TestCleaners")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("PoolSource",
    skipEvents = cms.untracked.uint32(182),
    fileNames = cms.untracked.vstring('file:in.root')
)

process.cleanMuons = cms.EDFilter("PATMuonCleaner",
    muonSource = cms.InputTag("muons")
)

process.cleanElectrons = cms.EDFilter("PATElectronCleaner",
    removeDuplicates = cms.bool(True),
    electronSource = cms.InputTag("pixelMatchGsfElectrons")
)

process.cleanJets = cms.EDFilter("PATCaloJetCleaner",
    removeOverlaps = cms.VPSet(cms.PSet(
        deltaR = cms.double(0.3),
        collection = cms.InputTag("cleanElectrons")
    )),
    jetSource = cms.InputTag("iterativeCone5CaloJets")
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoCaloJets_*_*_*', 
        'keep pixelMatchGsfElectrons_*_*_*', 
        'keep *_*_*_TestCleaners'),
    fileName = cms.untracked.string('/tmp/gpetrucc/out.root')
)

process.p = cms.Path(process.cleanElectrons*process.cleanJets)
process.e = cms.EndPath(process.out)


