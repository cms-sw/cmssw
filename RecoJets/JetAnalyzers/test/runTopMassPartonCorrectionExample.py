import FWCore.ParameterSet.Config as cms

process = cms.Process("TopMass")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

#  PYTHIA6 Generator 
process.load("Configuration.Generator.TTbar_cfi")

process.myPartons = cms.EDFilter("PartonSelector",
    withLeptons = cms.bool(False)
)

process.matchParton = cms.EDFilter("JetPartonMatcher",
    jets = cms.InputTag("iterativeCone5GenJets"),
    coneSizeToAssociate = cms.double(0.3),
    partons = cms.InputTag("myPartons")
)

process.doTopMass = cms.EDFilter("calcTopMass",
    srcSelectedPartons = cms.InputTag("myPartons"),
    srcByReference = cms.InputTag("matchParton"),
)

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.generator*process.pgen*process.myPartons*process.matchParton*process.printEventNumber)
process.outpath = cms.EndPath(process.doTopMass)
process.MessageLogger.destinations = cms.untracked.vstring('cout','cerr')


