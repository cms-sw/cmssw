import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V6::All'

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.demo2 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('JetCorrectorParametersCollection_Spring10_AK5Calo'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(False)
)


process.demo3 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('JetCorrectorParametersCollection_Spring10_AK5JPT'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(False)
)

process.p = cms.Path(process.demo2 * process.demo3)
