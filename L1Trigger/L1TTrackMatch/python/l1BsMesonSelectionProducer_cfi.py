import FWCore.ParameterSet.Config as cms

l1BsMesonSelectionProducer = cms.EDProducer('L1BsMesonSelectionProducer',
  l1PhiCandInputTag = cms.InputTag("l1PhiMesonSelectionProducer", "Level1PhiMesonColl"),
  outputCollectionName = cms.string("Level1BsHadronicColl"),
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

