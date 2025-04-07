import FWCore.ParameterSet.Config as cms

l1BsMesonSelectionEmulationProducer = cms.EDProducer('L1BsMesonSelectionEmulationProducer',
  l1PhiMesonWordInputTag = cms.InputTag("l1PhiMesonSelectionEmulationProducer","Level1PhiMesonEmulationColl"),
  posTrackInputTag = cms.InputTag("l1KaonTrackSelectionProducer", "Level1TTKaonTracksSelectedEmulationPositivecharge"),
  negTrackInputTag = cms.InputTag("l1KaonTrackSelectionProducer", "Level1TTKaonTracksSelectedEmulationNegativecharge"),
  outputCollectionName = cms.string("Level1BsHadronicEmulationColl"),
  cutSet = cms.PSet(
      phiPairdzMax = cms.double(0.5),
      phiPairdRMin = cms.double(0.2),
      phiPairdRMax = cms.double(1.0),
      phiPairMMin = cms.double(5.0),
      phiPairMMax = cms.double(5.8),
      bsPtMin = cms.double(13.0)
  ),
  debug = cms.int32(0)
)
