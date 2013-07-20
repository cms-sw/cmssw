import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring('file:patTuple.root'), ## mandatory
    maxEvents   = cms.int32(100),                            ## optional
    outputEvery = cms.uint32(10),                            ## optional
)
    
process.fwliteOutput = cms.PSet(
    fileName  = cms.string('analyzeFWLiteHistograms.root'),  ## mandatory
)

process.btagAnalyzer = cms.PSet(
    ## input specific for this analyzer
 Jets = cms.InputTag("cleanPatJets"),    
   bTagAlgo=cms.string('trackCountingHighEffBJetTags'),
   bins=cms.uint32(100),
   lowerbin=cms.double(0.),
   upperbin=cms.double(10.)
)
process.jecAnalyzer = cms.PSet(
    ## input specific for this analyzer
      Jets = cms.InputTag("cleanPatJets"),
  jecLevel=cms.string("L2Relative"),
  patJetCorrFactors= cms.string('patJetCorrFactors')
)
