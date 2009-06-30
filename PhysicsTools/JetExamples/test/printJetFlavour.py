import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/5/20/RelVal-RelValTTbar-1211209682-FakeConditions-2nd/0000/08765709-5826-DD11-9CE8-000423D94700.root')
)

process.myPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False)
)

process.flavourByRef = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("iterativeCone5CaloJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

process.flavourByVal = cms.EDFilter("JetFlavourIdentifier",
    srcByReference = cms.InputTag("flavourByRef"),
    physicsDefinition = cms.bool(False)
)

process.printEvent = cms.EDFilter("printJetFlavour",
    srcSelectedPartons = cms.InputTag("myPartons"),
    srcByReference = cms.InputTag("flavourByRef"),
    srcByValue = cms.InputTag("flavourByVal")
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.myPartons*process.flavourByRef*process.flavourByVal*process.printEvent)
process.outpath = cms.EndPath(process.printEventNumber)
process.MessageLogger.destinations = ['cout']
process.MessageLogger.cout = cms.PSet(
    threshold = cms.untracked.string('ERROR')
)


