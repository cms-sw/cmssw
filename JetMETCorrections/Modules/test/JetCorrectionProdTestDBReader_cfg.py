import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_38Y_V13::All'

from JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff import *



process.GlobalTag.toGet = cms.VPSet(
    cms.PSet(record = cms.string("JetCorrectionsRecord"),
             tag = cms.string("JetCorrectorParametersCollection_Spring10_V5_AK5Calo"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
             label=cms.untracked.string("AK5CaloNew")),
    cms.PSet(record = cms.string("JetCorrectionsRecord"),
             tag = cms.string("JetCorrectorParametersCollection_Summer10_V5_AK5JPT"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
             label=cms.untracked.string("AK5JPTNew")),
    cms.PSet(record = cms.string("JetCorrectionsRecord"),
             tag = cms.string("JetCorrectorParametersCollection_Spring10_V5_IC5Calo"),
             connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS"),
             label=cms.untracked.string("IC5CaloNew")),    
    )



process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.demo2 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK5CaloNew'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(False)
)


process.demo3 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('AK5JPTNew'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(False)
)


process.demo4 = cms.EDAnalyzer('JetCorrectorDBReader', 
        payloadName    = cms.untracked.string('IC5CaloNew'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(False)
)


process.p = cms.Path(process.demo2 * process.demo3 * process.demo4)
