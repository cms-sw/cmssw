import FWCore.ParameterSet.Config as cms

ssdigimultvtxposcorr = cms.EDAnalyzer('MultiplicityInvestigator',
                                      vertexCollection = cms.InputTag(""),
                                      wantInvestHist = cms.bool(False),
                                      wantVtxCorrHist = cms.bool(False),
                                      wantLumiCorrHist = cms.bool(False),
                                      wantPileupCorrHist = cms.bool(False),
                                      wantVtxPosCorrHist = cms.bool(True),
                                      digiVtxPosCorrConfig = cms.PSet(
    mcVtxCollection=cms.InputTag("generatorSmeared"),
    wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712/64))
    ),
    hitName = cms.untracked.string("digi"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(100)
    ),
                                      multiplicityMap = cms.InputTag("ssdigimultprod"),
                                    )

