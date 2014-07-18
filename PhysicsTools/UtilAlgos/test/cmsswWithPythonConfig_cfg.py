import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_7_1_0_pre1/RelValProdTTbar/GEN-SIM-RECO/START70_V5-v1/00000/14842A6B-2086-E311-B5CB-02163E00E8DA.root'
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

process.MessageLogger = cms.Service("MessageLogger")

## ---
## This is an example of the use of the BasicAnalyzer concept used to exploit C++ classes to do anaysis
## in full framework or FWLite using the same class. You can find the implementation of this module in
## PhysicsTools/UtilAlgos/plugins/WrappedEDMuonAnlyzer. You can find the EDAnalyzerWrapper.h class in
## PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h. You can find the implementation of the
## BasicMuonAnalyzer class in PhysicsTools/UtilAlgos/interface/BasicMuonAnlyzer.h. You will also find
## back the input parameters to the module.
process.muonAnalyzer = cms.EDAnalyzer("WrappedEDMuonAnalyzer",
  muons = cms.InputTag("muons"),
)

process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzeCMSSWHistograms.root')
)

process.p = cms.Path(process.muonAnalyzer)

