import FWCore.ParameterSet.Config as cms

def modify_hltL3TrajSeedOIHit(_hltL3TrajSeedOIHit):
    _iterativeTSG = _hltL3TrajSeedOIHit.TkSeedGenerator.iterativeTSG
    _iterativeTSG.ComponentName = cms.string('FastTSGFromPropagation')
    _iterativeTSG.HitProducer = cms.InputTag("fastMatchedTrackerRecHitCombinations")
    _iterativeTSG.MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent")
    _iterativeTSG.SimTrackCollectionLabel = cms.InputTag("fastSimProducer")
    _iterativeTSG.beamSpot = cms.InputTag("offlineBeamSpot")
    _hltL3TrajSeedOIHit.TrackerSeedCleaner = cms.PSet()

def modify_hltL3TrajSeedIOHit(_hltL3TrajSeedIOHit):
    _iterativeTSG = cms.PSet()
    _iterativeTSG.ComponentName = cms.string('FastTSGFromIOHit')
    _iterativeTSG.PtCut = cms.double(1.0)
    _iterativeTSG.SeedCollectionLabels = cms.VInputTag(
        cms.InputTag("initialStepSeeds"), 
        cms.InputTag("detachedTripletStepSeeds"), 
        cms.InputTag("lowPtTripletStepSeeds"), 
        cms.InputTag("pixelPairStepSeeds"))
    _iterativeTSG.SimTrackCollectionLabel = cms.InputTag("fastSimProducer")
    _hltL3TrajSeedIOHit.TkSeedGenerator.iterativeTSG = _iterativeTSG
    _hltL3TrajSeedIOHit.TrackerSeedCleaner = cms.PSet()
    

