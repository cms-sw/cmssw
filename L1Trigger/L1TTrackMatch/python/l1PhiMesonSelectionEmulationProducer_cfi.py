import FWCore.ParameterSet.Config as cms

l1PhiMesonSelectionEmulationProducer = cms.EDProducer('L1PhiMesonSelectionEmulationProducer',
  l1PosKaonTracksInputTag = cms.InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedEmulationPositivecharge"),
  l1NegKaonTracksInputTag = cms.InputTag("l1KaonTrackSelectionProducer","Level1TTKaonTracksSelectedEmulationNegativecharge"),
  outputCollectionName = cms.string("Level1PhiMesonEmulationColl"),
  cutSet = cms.PSet(
      tkPairdzMax = cms.double(0.5),
      tkPairdRMax = cms.double(0.2),
      tkPairMMin = cms.double(1.0), 
      tkPairMMax = cms.double(1.03) 
  ),
  debug = cms.int32(0)
)
