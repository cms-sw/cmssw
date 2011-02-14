### Open-HLT includes, modules and paths for b-jet
import copy
import FWCore.ParameterSet.Config as cms

# import the whole HLT menu
#from HLTrigger.Configuration.HLT_8E29_cff import *
#from HLTrigger.Configuration.HLT_1E31_cff import *
from HLTrigger.Configuration.HLT_FULL_cff import *

### lifetime-based b-tag OpenHLT (ideal) ######################################
# L1 filter is skipped
# L2 reco sequence is common to all paths, and taken from the global table
# L2.5 and L3 sequences are rewritten to bypass selectors and filters

# L2.5 reco modules (common with strt up)

# BJetinputjetCollection="hltIterativeCone5CaloJets"
BJetinputjetCollection="hltAntiKT5CaloJets"

openHltBLifetimeL25Associator = copy.deepcopy(hltBLifetimeL25Associator)
openHltBLifetimeL25Associator.jets = cms.InputTag(BJetinputjetCollection)

openHltBLifetimeL25TagInfos = copy.deepcopy(hltBLifetimeL25TagInfos)
openHltBLifetimeL25TagInfos.jetTracks = cms.InputTag("openHltBLifetimeL25Associator")

openHltBLifetimeL25BJetTags = copy.deepcopy(hltBLifetimeL25BJetTags)
openHltBLifetimeL25BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL25TagInfos"))

# L2.5 reco sequence for lifetime tagger
OpenHLTBLifetimeL25recoSequence = cms.Sequence(
        HLTDoLocalPixelSequence +
        HLTRecopixelvertexingSequence +
        openHltBLifetimeL25Associator +
        openHltBLifetimeL25TagInfos +
        openHltBLifetimeL25BJetTags )

# L3 reco modules

openHltBLifetimeRegionalPixelSeedGenerator = copy.deepcopy(hltBLifetimeRegionalPixelSeedGenerator)
openHltBLifetimeRegionalPixelSeedGenerator.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag(BJetinputjetCollection)

openHltBLifetimeRegionalCkfTrackCandidates = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidates)
openHltBLifetimeRegionalCkfTrackCandidates.src = cms.InputTag("openHltBLifetimeRegionalPixelSeedGenerator")

openHltBLifetimeRegionalCtfWithMaterialTracks = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracks)
openHltBLifetimeRegionalCtfWithMaterialTracks.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidates")

openHltBLifetimeL3Associator = copy.deepcopy(hltBLifetimeL3Associator)
openHltBLifetimeL3Associator.jets   = cms.InputTag(BJetinputjetCollection)
openHltBLifetimeL3Associator.tracks = cms.InputTag("openHltBLifetimeRegionalCtfWithMaterialTracks")

openHltBLifetimeL3TagInfos = copy.deepcopy(hltBLifetimeL3TagInfos)
openHltBLifetimeL3TagInfos.jetTracks = cms.InputTag("openHltBLifetimeL3Associator")

openHltBLifetimeL3BJetTags = copy.deepcopy(hltBLifetimeL3BJetTags)
openHltBLifetimeL3BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfos"))

# L3 reco sequence for lifetime tagger
OpenHLTBLifetimeL3recoSequence = cms.Sequence(
    HLTDoLocalPixelSequence +
    HLTDoLocalStripSequence +
    openHltBLifetimeRegionalPixelSeedGenerator +
    openHltBLifetimeRegionalCkfTrackCandidates +
    openHltBLifetimeRegionalCtfWithMaterialTracks +
    openHltBLifetimeL3Associator +
    openHltBLifetimeL3TagInfos +
    openHltBLifetimeL3BJetTags )

    ### lifetime-based b-tag OpenHLT (start up) ###################################
# L1 filter is skipped
# L2 reco sequence is common to all paths, and taken from the global table
# L2.5 as rewritten is common with ideal conditions
# L3 sequence is rewritten to bypass selectors and filters

openHltBLifetimeRegionalPixelSeedGeneratorStartup = copy.deepcopy(hltBLifetimeRegionalPixelSeedGenerator)
openHltBLifetimeRegionalPixelSeedGeneratorStartup.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag(BJetinputjetCollection)

openHltBLifetimeRegionalCkfTrackCandidatesStartup = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidates)
openHltBLifetimeRegionalCkfTrackCandidatesStartup.src = cms.InputTag("openHltBLifetimeRegionalPixelSeedGeneratorStartup")

openHltBLifetimeRegionalCtfWithMaterialTracksStartup = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracks)
openHltBLifetimeRegionalCtfWithMaterialTracksStartup.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidatesStartup")

openHltBLifetimeL3AssociatorStartup = copy.deepcopy(hltBLifetimeL3Associator)
openHltBLifetimeL3AssociatorStartup.jets   = cms.InputTag(BJetinputjetCollection)
openHltBLifetimeL3AssociatorStartup.tracks = cms.InputTag("openHltBLifetimeRegionalCtfWithMaterialTracksStartup")

openHltBLifetimeL3TagInfosStartup = copy.deepcopy(hltBLifetimeL3TagInfos)
openHltBLifetimeL3TagInfosStartup.jetTracks = cms.InputTag("openHltBLifetimeL3AssociatorStartup")

openHltBLifetimeL3BJetTagsStartup = copy.deepcopy(hltBLifetimeL3BJetTags)
openHltBLifetimeL3BJetTagsStartup.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfosStartup"))

# L3 reco sequence for relaxed lifetime tagger
OpenHLTBLifetimeL3recoSequenceStartup = cms.Sequence(
    HLTDoLocalPixelSequence +
    HLTDoLocalStripSequence +
    openHltBLifetimeRegionalPixelSeedGeneratorStartup +
    openHltBLifetimeRegionalCkfTrackCandidatesStartup +
    openHltBLifetimeRegionalCtfWithMaterialTracksStartup +
    openHltBLifetimeL3AssociatorStartup +
    openHltBLifetimeL3TagInfosStartup +
    openHltBLifetimeL3BJetTagsStartup )

### soft-muon-based b-tag OpenHLT (ideal, start up, and performance meas.) ####
# L1 filter is skipped
# L2 reco sequence is common to all paths, and taken from the global table
# L2.5 and L3 sequences are rewritten to bypass selectors and filters

openHltBSoftmuonL25TagInfos = copy.deepcopy(hltBSoftMuonL25TagInfos)
openHltBSoftmuonL25TagInfos.jets = cms.InputTag(BJetinputjetCollection)

openHltBSoftmuonL25BJetTags = copy.deepcopy(hltBSoftMuonL25BJetTagsByDR)
openHltBSoftmuonL25BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL25TagInfos"))

openHltBSoftmuonL3TagInfos = copy.deepcopy(hltBSoftMuon5SelL3TagInfos)
openHltBSoftmuonL3TagInfos.jets = cms.InputTag(BJetinputjetCollection)

openHltBSoftmuonL3BJetTags = copy.deepcopy(hltBSoftMuon5SelL3BJetTagsByPt)
openHltBSoftmuonL3BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL3TagInfos"))

openHltBPerfMeasL3BJetTags = copy.deepcopy(hltBSoftMuon5SelL3BJetTagsByDR)
openHltBPerfMeasL3BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL3TagInfos"))

OpenHLTBSoftMuonL25recoSequence = cms.Sequence(
    HLTL2muonrecoNocandSequence +
    openHltBSoftmuonL25TagInfos +
    openHltBSoftmuonL25BJetTags )

OpenHLTBSoftMuonL3recoSequence = cms.Sequence(
    HLTL3muonrecoNocandSequence +
    hltBSoftMuon5L3 +
    openHltBSoftmuonL3TagInfos +
    openHltBSoftmuonL3BJetTags +
    openHltBPerfMeasL3BJetTags )
