import FWCore.ParameterSet.Config as cms

import FastSimulation.HighLevelTrigger.DummyModule_cfi
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#############################################
# Reconstruct tracks with pixel seeds
#############################################

# Take all pixel tracks for b tagging track reco (pTMin>1GeV, nHits>=8)
hltFastTrackMergerForFastJet = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0),
    minHits = cms.untracked.uint32(8)
)

hltDisplacedHT250L1FastJetRegionalPixelSeedGenerator = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltDisplacedHT300L1FastJetRegionalPixelSeedGenerator = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()

hltDisplacedHT250L1FastJetRegionalCkfTrackCandidates = cms.Sequence(globalPixelTracking)
hltDisplacedHT300L1FastJetRegionalCkfTrackCandidates  = cms.Sequence(globalPixelTracking)

hltDisplacedHT250L1FastJetRegionalCtfWithMaterialTracks = hltFastTrackMergerForFastJet.clone()
hltDisplacedHT300L1FastJetRegionalCtfWithMaterialTracks = hltFastTrackMergerForFastJet.clone() 
