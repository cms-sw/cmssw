import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import *
pixelTripletSeedsForMinBias = copy.deepcopy(trajectorySeedProducer)
import copy
from FastSimulation.Tracking.PixelTracksProducer_cfi import *
pixelTracksForMinBias = copy.deepcopy(pixelTracks)
from FastSimulation.Tracking.PixelTracksProducer_cfi import *
import copy
from HLTrigger.HLTfilters.hltBool_cfi import *
filterTriggerType = copy.deepcopy(hltBool)
pixelTrackingForMinBias = cms.Sequence(pixelTripletSeedsForMinBias*pixelTracksForMinBias)
pixelTrackingForIsol = cms.Sequence(cms.SequencePlaceholder("pixelGSTracking"))
l1sEcalPhiSym.L1MuonCollectionTag = 'l1ParamMuons'
l1sHcalPhiSym.L1MuonCollectionTag = 'l1ParamMuons'
l1sEcalPi0.L1MuonCollectionTag = 'l1ParamMuons'
l1seedMinBiasPixel.L1MuonCollectionTag = 'l1ParamMuons'
hltl1sMin.L1MuonCollectionTag = 'l1ParamMuons'
hltl1sZero.L1MuonCollectionTag = 'l1ParamMuons'
l1sIsolTrack.L1MuonCollectionTag = 'l1ParamMuons'
level1seedHLTBackwardBSC.L1MuonCollectionTag = 'l1ParamMuons'
level1seedHLTForwardBSC.L1MuonCollectionTag = 'l1ParamMuons'
level1seedHLTCSCBeamHalo.L1MuonCollectionTag = 'l1ParamMuons'
level1seedHLTCSCBeamHaloOverlapRing1.L1MuonCollectionTag = 'l1ParamMuons'
level1seedHLTCSCBeamHaloOverlapRing2.L1MuonCollectionTag = 'l1ParamMuons'
level1seedHLTCSCBeamHaloRing2or3.L1MuonCollectionTag = 'l1ParamMuons'
l1sEcalPhiSym.L1GtObjectMapTag = 'gtDigis'
l1sHcalPhiSym.L1GtObjectMapTag = 'gtDigis'
l1sEcalPi0.L1GtObjectMapTag = 'gtDigis'
l1seedMinBiasPixel.L1GtObjectMapTag = 'gtDigis'
hltl1sMin.L1GtObjectMapTag = 'gtDigis'
hltl1sZero.L1GtObjectMapTag = 'gtDigis'
l1sIsolTrack.L1GtObjectMapTag = 'gtDigis'
level1seedHLTBackwardBSC.L1GtObjectMapTag = 'gtDigis'
level1seedHLTForwardBSC.L1GtObjectMapTag = 'gtDigis'
level1seedHLTCSCBeamHalo.L1GtObjectMapTag = 'gtDigis'
level1seedHLTCSCBeamHaloOverlapRing1.L1GtObjectMapTag = 'gtDigis'
level1seedHLTCSCBeamHaloOverlapRing2.L1GtObjectMapTag = 'gtDigis'
level1seedHLTCSCBeamHaloRing2or3.L1GtObjectMapTag = 'gtDigis'
pixelTripletSeedsForMinBias.numberOfHits = [3]
pixelTripletSeedsForMinBias.firstHitSubDetectorNumber = [2]
pixelTripletSeedsForMinBias.firstHitSubDetectors = [1, 2]
pixelTripletSeedsForMinBias.secondHitSubDetectorNumber = [2]
pixelTripletSeedsForMinBias.secondHitSubDetectors = [1, 2]
pixelTripletSeedsForMinBias.thirdHitSubDetectorNumber = [2]
pixelTripletSeedsForMinBias.thirdHitSubDetectors = [1, 2]
pixelTripletSeedsForMinBias.seedingAlgo = ['PixelTriplet']
pixelTripletSeedsForMinBias.originpTMin = [0.2]
pixelTripletSeedsForMinBias.pTMin = [0.2]
pixelTracksForMinBias.SeedProducer = cms.InputTag("pixelTripletSeedsForMinBias","PixelTriplet")

