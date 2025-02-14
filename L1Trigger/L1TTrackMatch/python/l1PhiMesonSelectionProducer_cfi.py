import FWCore.ParameterSet.Config as cms

l1PhiMesonSelectionProducer = cms.EDProducer('L1PhiMesonSelectionProducer',
  l1PosKaonTracksInputTag = cms.InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedPositivecharge"),
  l1NegKaonTracksInputTag = cms.InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedNegativecharge"),
  outputCollectionName = cms.string("Level1PhiMesonColl"),
  cutSet = cms.PSet(
      tkPairdzMax = cms.double(0.5),
      tkPairdRMax = cms.double(0.2),
      tkPairMMin = cms.double(1.0), 
      tkPairMMax = cms.double(1.03) 
  ),
  debug = cms.int32(0)
)


