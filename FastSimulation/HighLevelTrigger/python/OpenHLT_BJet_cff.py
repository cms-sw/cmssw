### Open-HLT includes, modules and paths for b-jet
import copy
import FWCore.ParameterSet.Config as cms

# import the whole HLT menu
from FastSimulation.Configuration.HLT_cff import *
  
### lifetime-based b-tag OpenHLT (ideal) ######################################
# L1 filter is skipped
# L2 reco sequence is common to all paths, and taken from the global table
# L2.5 and L3 sequences are rewritten to bypass selectors and filters

# L2.5 reco modules (common with strt up)

openHltBLifetimeL25Associator = copy.deepcopy(hltBLifetimeL25AssociatorStartupU)

openHltBLifetimeL25TagInfos = copy.deepcopy(hltBLifetimeL25TagInfosStartupU)
openHltBLifetimeL25TagInfos.jetTracks = cms.InputTag("openHltBLifetimeL25Associator")

openHltBLifetimeL25BJetTags = copy.deepcopy(hltBLifetimeL25BJetTagsStartupU)
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
openHltBLifetimeRegionalPixelSeedGenerator.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("hltIterativeCone5CaloJets")

openHltBLifetimeRegionalCkfTrackCandidates = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidates)
openHltBLifetimeRegionalCkfTrackCandidates.src = cms.InputTag("openHltBLifetimeRegionalPixelSeedGenerator")

openHltBLifetimeRegionalCtfWithMaterialTracks = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracks)
openHltBLifetimeRegionalCtfWithMaterialTracks.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidates")
  
openHltBLifetimeL3Associator = copy.deepcopy(hltBLifetimeL3Associator)
openHltBLifetimeL3Associator.jets   = cms.InputTag("hltIterativeCone5CaloJets")
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

openHltBLifetimeRegionalPixelSeedGeneratorRelaxed = copy.deepcopy(hltBLifetimeRegionalPixelSeedGeneratorRelaxed)
openHltBLifetimeRegionalPixelSeedGeneratorRelaxed.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("hltIterativeCone5CaloJets")

openHltBLifetimeRegionalCkfTrackCandidatesRelaxed = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidatesRelaxed)
openHltBLifetimeRegionalCkfTrackCandidatesRelaxed.src = cms.InputTag("openHltBLifetimeRegionalPixelSeedGeneratorRelaxed")

openHltBLifetimeRegionalCtfWithMaterialTracksRelaxed = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracksRelaxed)
openHltBLifetimeRegionalCtfWithMaterialTracksRelaxed.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidatesRelaxed")
  
openHltBLifetimeL3AssociatorRelaxed = copy.deepcopy(hltBLifetimeL3AssociatorRelaxed)
openHltBLifetimeL3AssociatorRelaxed.jets   = cms.InputTag("hltIterativeCone5CaloJets")
openHltBLifetimeL3AssociatorRelaxed.tracks = cms.InputTag("openHltBLifetimeRegionalCtfWithMaterialTracksRelaxed")

openHltBLifetimeL3TagInfosRelaxed = copy.deepcopy(hltBLifetimeL3TagInfosRelaxed)
openHltBLifetimeL3TagInfosRelaxed.jetTracks = cms.InputTag("openHltBLifetimeL3AssociatorRelaxed")
  
openHltBLifetimeL3BJetTagsRelaxed = copy.deepcopy(hltBLifetimeL3BJetTagsRelaxed)
openHltBLifetimeL3BJetTagsRelaxed.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfosRelaxed"))
  
# L3 reco sequence for relaxed lifetime tagger
OpenHLTBLifetimeL3recoSequenceRelaxed = cms.Sequence( 
    HLTDoLocalPixelSequence + 
    HLTDoLocalStripSequence + 
    openHltBLifetimeRegionalPixelSeedGeneratorRelaxed + 
    openHltBLifetimeRegionalCkfTrackCandidatesRelaxed + 
    openHltBLifetimeRegionalCtfWithMaterialTracksRelaxed + 
    openHltBLifetimeL3AssociatorRelaxed + 
    openHltBLifetimeL3TagInfosRelaxed + 
    openHltBLifetimeL3BJetTagsRelaxed )

### soft-muon-based b-tag OpenHLT (ideal, start up, and performance meas.) ####
# L1 filter is skipped
# L2 reco sequence is common to all paths, and taken from the global table
# L2.5 and L3 sequences are rewritten to bypass selectors and filters

openHltBSoftmuonL25TagInfos = copy.deepcopy(hltBSoftmuonL25TagInfos)
openHltBSoftmuonL25TagInfos.jets = cms.InputTag("hltIterativeCone5CaloJets")

openHltBSoftmuonL25BJetTags = copy.deepcopy(hltBSoftmuonL25BJetTags)
openHltBSoftmuonL25BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL25TagInfos"))

openHltBSoftmuonL3TagInfos = copy.deepcopy(hltBSoftmuonL3TagInfos)
openHltBSoftmuonL3TagInfos.jets = cms.InputTag("hltIterativeCone5CaloJets")

openHltBSoftmuonL3BJetTags = copy.deepcopy(hltBSoftmuonL3BJetTags)
openHltBSoftmuonL3BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL3TagInfos"))

openHltBPerfMeasL3BJetTags = copy.deepcopy(hltBSoftmuonL3BJetTagsByDR)
openHltBPerfMeasL3BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL3TagInfos"))

OpenHLTBSoftmuonL25recoSequence = cms.Sequence( 
    HLTL2muonrecoNocandSequence +
    openHltBSoftmuonL25TagInfos +
    openHltBSoftmuonL25BJetTags ) 

OpenHLTBSoftmuonL3recoSequence = cms.Sequence( 
    HLTL3muonrecoNocandSequence +
    openHltBSoftmuonL3TagInfos +
    openHltBSoftmuonL3BJetTags +
    openHltBPerfMeasL3BJetTags )

