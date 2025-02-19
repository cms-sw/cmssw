import FWCore.ParameterSet.Config as cms

process = cms.Process("TopMass")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)

# PYTHIA6 Generator 
process.load("Configuration.Generator.TTbar_cfi")

# Parton Correction
process.load("JetMETCorrections.Configuration.L7PartonCorrections_cff")

process.myPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False)
)

process.matchParton = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("sisCone5GenJets"),
    coneSizeToAssociate = cms.double(0.10),
    partons = cms.InputTag("myPartons")
)

process.doTopMass = cms.EDFilter("calcTopMass",
    srcByReference = cms.InputTag("matchParton"),
    qTopCorrector = cms.string('L7PartonJetCorrectorSC5qTop'),
    cTopCorrector = cms.string('L7PartonJetCorrectorSC5cTop'),
    bTopCorrector = cms.string('L7PartonJetCorrectorSC5bTop'),
    tTopCorrector = cms.string('L7PartonJetCorrectorSC5tTop')
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("histoMass.root")
 )

process.p = cms.Path(process.generator*process.pgen*process.myPartons*process.matchParton*process.printEventNumber)
process.outpath = cms.EndPath(process.doTopMass)
process.MessageLogger.destinations = cms.untracked.vstring('cout','cerr')


