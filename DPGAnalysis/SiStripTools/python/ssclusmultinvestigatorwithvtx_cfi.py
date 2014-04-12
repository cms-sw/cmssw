import FWCore.ParameterSet.Config as cms

ssclusmultinvestigatorwithvtx = cms.EDAnalyzer('MultiplicityInvestigator',
                                               vertexCollection = cms.InputTag("offlinePrimaryVertices"),
                                               wantInvestHist = cms.bool(True),
                                               wantVtxCorrHist = cms.bool(True),
                                               wantLumiCorrHist = cms.bool(False),
                                               wantPileupCorrHist = cms.bool(False),
                                               wantVtxPosCorrHist = cms.bool(False),
                                               digiVtxCorrConfig = cms.PSet(
    wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712/64))
    ),
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(10)
    ),
                                               wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712/64)),
    cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/64)),
    cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID"), binMax = cms.int32( 565248/64)),
    cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB"), binMax = cms.int32(3303936/64)),
    cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"), binMax = cms.int32(3866624/64))
    ),
                                               hitName = cms.untracked.string("cluster"),
                                               multiplicityMap = cms.InputTag("ssclustermultprod"),
                                               numberOfBins = cms.untracked.int32(500),   
                                               maxLSBeforeRebin = cms.untracked.uint32(100),   
                                               startingLSFraction = cms.untracked.uint32(4),   
                                               scaleFactor = cms.untracked.int32(10)
                                               )

