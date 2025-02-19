import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.Generator.PythiaUESettings_cfi")

process.load("Configuration.StandardSequences.Generator_cff")

process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)
process.source = cms.Source("PythiaSource",
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        processParameters = cms.vstring('MSEL=5               ! high pT process -> bb', 
            'CKIN(3)=170.         ! minimum pt hat for hard interactions', 
            'CKIN(4)=230.         ! maximum pt hat for hard interactions'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'processParameters')
    )
)

process.printTree = cms.EDFilter("ParticleListDrawer",
    src = cms.InputTag("genParticles"),
    maxEventsToPrint = cms.untracked.int32(1)
)

process.BCRatio = cms.EDFilter("GenJetBCEnergyRatio",
    genJets = cms.InputTag("iterativeCone5GenJets")
)

process.printEvent = cms.EDAnalyzer("printGenJetRatio",
    srcBratio = cms.InputTag("BCRatio","bRatioCollection"),
    srcCratio = cms.InputTag("BCRatio","cRatioCollection")
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.pgen*process.printTree*process.BCRatio*process.printEvent)
process.outpath = cms.EndPath(process.printEventNumber)
process.MessageLogger.cerr.default.limit = 10


