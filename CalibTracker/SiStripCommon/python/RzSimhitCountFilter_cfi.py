import FWCore.ParameterSet.Config as cms

rzSimhitCountFilter = cms.EDFilter("RzSimhitCountFilter",
                                   MaxRadius = cms.double(1000), #Much bigger than actual tracker
                                   MaxZ = cms.double(1000),      #Much bigger than actual tracker
                                   MinHits = cms.uint32(4),
                                   InputTags = cms.VInputTag(
    cms.InputTag('g4SimHits:TrackerHitsTECHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTECLowTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIDHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIDLowTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIBHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTIBLowTof'),
    cms.InputTag('g4SimHits:TrackerHitsTOBHighTof'),
    cms.InputTag('g4SimHits:TrackerHitsTOBLowTof')
    ))
