import FWCore.ParameterSet.Config as cms

ssdigimultinvestigatorwithvtx = cms.EDAnalyzer('MultiplicityInvestigator',
                                               vertexCollection = cms.InputTag("offlinePrimaryVertices"),
                                               wantInvestHist = cms.bool(True),
                                               wantVtxCorrHist = cms.bool(True),
                                               wantLumiCorrHist = cms.bool(False),
                                               wantPileupCorrHist = cms.bool(False),
                                               digiLumiCorrConfig = cms.PSet(lumiProducer=cms.InputTag("")),
                                               digiPileupCorrConfig = cms.PSet(
                                                                               pileupSummaryCollection=cms.InputTag(""),
                                                                               useVisibleVertices = cms.bool(False)
                                                                               ),
                                               digiVtxCorrConfig = cms.PSet(
    wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712))
    ),
    hitName = cms.untracked.string("digi"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(100)
    ),
                                               wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712)),
    cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB"), binMax = cms.int32(1787904)),
    cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID"), binMax = cms.int32( 565248)),
    cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB"), binMax = cms.int32(3303936)),
    cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"), binMax = cms.int32(3866624))
    ),
                                               hitName = cms.untracked.string("digi"),
                                               multiplicityMap = cms.InputTag("ssdigimultprod"),
                                               numberOfBins = cms.untracked.int32(2000),   
                                               maxLSBeforeRebin = cms.untracked.uint32(100),   
                                               startingLSFraction = cms.untracked.uint32(4),   
                                               scaleFactor = cms.untracked.int32(100)
                                               )

