import FWCore.ParameterSet.Config as cms

import FastSimulation.HighLevelTrigger.DummyModule_cfi
from FastSimulation.Tracking.GlobalPixelTracking_cff import *

#############################################
# Reconstruct tracks with pixel seeds
#############################################

# Take all pixel tracks for b tagging track reco (pTMin>1GeV, nHits>=8)
hltFastTrackMergerForB = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(1.0),
    minHits = cms.untracked.uint32(8)
)




###############################

hltBLifetimeRegionalCkfTrackCandidatesHbb = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorHbbVBF = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20 = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20 = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeRegionalPixelSeedGeneratorbbPhiL1FastJet = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()
hltBLifetimeFastRegionalPixelSeedGeneratorHbbVBF = FastSimulation.HighLevelTrigger.DummyModule_cfi.dummyModule.clone()

hltBLifetimeRegionalCkfTrackCandidatesHbb = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesHbbVBF = cms.Sequence(globalPixelTracking)
hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb = cms.Sequence(globalPixelTracking)
hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20Hbb = cms.Sequence(globalPixelTracking)
hltBLifetimeRegionalCkfTrackCandidatesbbPhiL1FastJet = cms.Sequence(globalPixelTracking)
hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet = cms.Sequence(globalPixelTracking)
hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet = cms.Sequence(globalPixelTracking)
hltBLifetimeFastRegionalCkfTrackCandidatesHbbVBF = cms.Sequence(globalPixelTracking)

hltBLifetimeRegionalCtfWithMaterialTracksHbb = hltFastTrackMergerForB.clone()
hltBLifetimeRegionalCtfWithMaterialTracksHbbVBF = hltFastTrackMergerForB.clone()
hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb = hltFastTrackMergerForB.clone()
hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20Hbb = hltFastTrackMergerForB.clone()
hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJet = hltFastTrackMergerForB.clone()
hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet = hltFastTrackMergerForB.clone()
hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet = hltFastTrackMergerForB.clone()
hltBLifetimeFastRegionalCtfWithMaterialTracksHbbVBF = hltFastTrackMergerForB.clone()


#############################################
# Reconstruct muons for MumuK
#############################################
import FWCore.ParameterSet.Config as cms

# Take all pixel-seeded tracks for b tagging track reco (pTMin>1GeV, nHits>=8) 
hltCtfWithMaterialTracksMumuk = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("globalPixelWithMaterialTracks"),
                                   cms.InputTag("globalPixelTrackCandidates")),
    ptMin = cms.untracked.double(3.0),
    minHits = cms.untracked.uint32(5)
)

# produce ChargedCandidates from tracks
hltMumukAllConeTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumuk"),
    particleType = cms.string('mu-')
)

hltCkfTrackCandidatesMumuk = cms.Sequence(cms.SequencePlaceholder("HLTL3muonrecoSequence"))


#############################################
# Reconstruct muons for JPsiToMumu
#############################################

# Take all pixel-seeded tracks for b tagging track reco (pTMin>1GeV, nHits>=8) 
hltCtfWithMaterialTracksMumu = cms.EDProducer("FastTrackMerger",
    SaveTracksOnly = cms.untracked.bool(True),
    TrackProducers = cms.VInputTag(cms.InputTag("hltL3Muons")),
    ptMin = cms.untracked.double(3.0),
    minHits = cms.untracked.uint32(5)
)

# produce ChargedCandidates from tracks
hltMuTracks = cms.EDProducer("ConcreteChargedCandidateProducer",
    src = cms.InputTag("hltCtfWithMaterialTracksMumu"),
    particleType = cms.string('mu-')
)

hltCkfTrackCandidatesMumu = cms.Sequence(cms.SequencePlaceholder("HLTL3muonrecoNocandSequence"))


