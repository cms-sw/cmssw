import FWCore.ParameterSet.Config as cms

spclusmultlumicorr = cms.EDAnalyzer('MultiplicityInvestigator',
                                    vertexCollection = cms.InputTag(""),
                                    wantInvestHist = cms.bool(False),
                                    wantVtxCorrHist = cms.bool(False),
                                    wantLumiCorrHist = cms.bool(True),
                                    wantPileupCorrHist = cms.bool(False),
                                    wantVtxPosCorrHist = cms.bool(False),
                                    digiLumiCorrConfig = cms.PSet(
    lumiProducer = cms.InputTag("lumiProducer"),
    wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("Pixel"), binMax = cms.int32(200000))
    ),
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),
    scaleFactor = cms.untracked.int32(100)
    ),
                                    multiplicityMap = cms.InputTag("spclustermultprod"),
                                    )

