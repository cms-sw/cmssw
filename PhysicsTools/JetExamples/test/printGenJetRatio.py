import FWCore.ParameterSet.Config as cms

process = cms.Process("testJET")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('IOMC.RandomEngine.IOMC_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic8TeVCollision_cfi')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
)

# Input source
process.source = cms.Source("EmptySource")

from Configuration.Generator.PythiaUEZ2starSettings_cfi import *
process.generator = cms.EDFilter("Pythia6GeneratorFilter",
    pythiaPylistVerbosity = cms.untracked.int32(0),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(8000.0),
    maxEventsToPrint = cms.untracked.int32(0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        processParameters = cms.vstring(
            'MSEL=5               ! high pT process -> bb',
            'CKIN(3)=170.         ! minimum pt hat for hard interactions',
            'CKIN(4)=230.         ! maximum pt hat for hard interactions'
        ),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings','processParameters')
    )
)

process.printList = cms.EDAnalyzer("ParticleListDrawer",
    src = cms.InputTag("genParticles"),
    maxEventsToPrint = cms.untracked.int32(1)
)

process.BCRatio = cms.EDProducer("GenJetBCEnergyRatio",
    genJets = cms.InputTag("ak5GenJets")
)

process.printEvent = cms.EDAnalyzer("printGenJetRatio",
    srcBratio = cms.InputTag("BCRatio","bRatioCollection"),
    srcCratio = cms.InputTag("BCRatio","cRatioCollection")
)

process.p = cms.Path(process.generator*process.pgen*process.printList*process.BCRatio*process.printEvent)

#process.MessageLogger.cout = cms.PSet(
#    enable = cms.untracked.bool(True),
#    threshold = cms.untracked.string('ERROR')
#)
