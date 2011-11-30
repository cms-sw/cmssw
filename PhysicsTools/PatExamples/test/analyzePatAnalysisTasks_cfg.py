import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")


process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    "file:patTuple.root"
  )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.MessageLogger = cms.Service("MessageLogger")

## ---
## This is an example of the use of the BasicAnalyzer concept used to exploit C++ classes to do anaysis
## in full framework or FWLite using the same class. You can find the implementation of this module in
## PhysicsTools/UtilAlgos/plugins/WrappedEDMuonAnlyzer. You can find the EDAnalyzerWrapper.h class in
## PhysicsTools/UtilAlgos/interface/EDAnalyzerWrapper.h. You can find the implementation of the
## BasicMuonAnalyzer class in PhysicsTools/UtilAlgos/interface/BasicMuonAnlyzer.h. You will also find
## back the input parameters to the module.
process.btagAnalyzer = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerBTag",
  Jets = cms.InputTag("cleanPatJets"),    
   bTagAlgo=cms.string('trackCountingHighEffBJetTags'),
   bins=cms.uint32(100),
   lowerbin=cms.double(0.),
   upperbin=cms.double(10.)
)
process.jecAnalyzer = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerJEC",
  Jets = cms.InputTag("cleanPatJets"),
  jecLevel=cms.string("L3Absolute"),
  patJetCorrFactors= cms.string('patJetCorrFactors')
)
process.jecAnalyzerRel = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerJEC",
  Jets = cms.InputTag("cleanPatJets"),
  jecLevel=cms.string("L2Relative"),
  patJetCorrFactors= cms.string('patJetCorrFactors')
)
process.jecAnalyzerNon = cms.EDAnalyzer("WrappedEDAnalysisTasksAnalyzerJEC",
  Jets = cms.InputTag("cleanPatJets"),
  jecLevel=cms.string("Uncorrected"),
  patJetCorrFactors= cms.string('patJetCorrFactors')
)
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('analyzeCMSSWHistograms.root')
)

process.p = cms.Path(process.btagAnalyzer * process.jecAnalyzer *process.jecAnalyzerRel *process.jecAnalyzerNon)

