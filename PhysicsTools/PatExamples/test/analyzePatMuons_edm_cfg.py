import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:patTuple.root"
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.MessageLogger = cms.Service("MessageLogger")

## ---
## This is an example of the use of the BasicAnalyzer concept used to exploit C++ classes to do anaysis
## in full framework or FWLite using the same class. You can find the implementation of this module in
## PhysicsTools/PatExamples/plugins/PatMuonEDAnlyzer.cc. You can find the EDAnalyzerWrapper.h class in
## PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h. You can find the implementation of the
## PatMuonAnalyzer class in PhysicsTools/PatExamples/interface/PatMuonAnlyzer.h. You will also find
## back the input parameters to the module.
process.patMuonAnalyzer = cms.EDAnalyzer("PatMuonEDAnalyzer",
  muons = cms.InputTag("cleanPatMuons"),                                             
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzePatMuons.root')
)

process.p = cms.Path(process.patMuonAnalyzer)

