### Open-HLT includes, modules and paths for b-jet
import copy
import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTanalyzers.HLT_FULL_cff import *

### lifetime-based b-tag OpenHLT ######################################
# L1 filter is skipped
# L2 reco sequence is common to all paths, and taken from the global table
# L2.5 and L3 sequences are rewritten to bypass selectors and filters

# L2.5 reco modules

###### Use corrected jets same as 5E32 Menu
BJetinputjetCollection="hltCaloJetCorrected"

openHltBLifetimeL25Associator = copy.deepcopy(hltBLifetimeL25AssociatorSingleTop)
openHltBLifetimeL25Associator.jets = cms.InputTag(BJetinputjetCollection)

openHltBLifetimeL25TagInfos = copy.deepcopy(hltBLifetimeL25TagInfosSingleTop)
openHltBLifetimeL25TagInfos.jetTracks = cms.InputTag("openHltBLifetimeL25Associator")

openHltBLifetimeL25BJetTags = copy.deepcopy(hltBLifetimeL25BJetTagsSingleTop)
openHltBLifetimeL25BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL25TagInfos"))

# Modules specific to Single Track TC
hltESPTrackCounting3D1st = cms.ESProducer( "TrackCountingESProducer",
                                           appendToDataLabel = cms.string( "" ),
                                           nthTrack = cms.int32( 1 ),
                                           impactParameterType = cms.int32( 0 ),
                                           deltaR = cms.double( -1.0 ),
                                           maximumDecayLength = cms.double( 5.0 ),
                                           maximumDistanceToJetAxis = cms.double( 0.07 ),
                                           trackQualityClass = cms.string( "any" )
                                        )


hltBLifetimeL25BJetTagsSingleTrack = cms.EDProducer( "JetTagProducer",
                                                     jetTagComputer = cms.string( "hltESPTrackCounting3D1st" ),
                                                     tagInfos = cms.VInputTag( 'hltBLifetimeL25TagInfosSingleTop' )
                                                     )


hltBLifetimeL3BJetTagsSingleTrack = cms.EDProducer( "JetTagProducer",
                                                    jetTagComputer = cms.string( "hltESPTrackCounting3D1st" ),
                                                    tagInfos = cms.VInputTag( 'hltBLifetimeL3TagInfosSingleTop' )
                                                    )

# Single Track TC
openHltBLifetimeL25BJetTagsSingleTrack = copy.deepcopy(hltBLifetimeL25BJetTagsSingleTrack)
openHltBLifetimeL25BJetTagsSingleTrack.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL25TagInfos"))

# L2.5 reco sequence for lifetime tagger
OpenHLTBLifetimeL25recoSequence = cms.Sequence(
        HLTDoLocalPixelSequence +
        HLTRecopixelvertexingSequence +
        openHltBLifetimeL25Associator +
        openHltBLifetimeL25TagInfos +
        openHltBLifetimeL25BJetTagsSingleTrack +
        openHltBLifetimeL25BJetTags )

# L3 reco modules
openHltBLifetimeRegionalPixelSeedGenerator = copy.deepcopy(hltBLifetimeRegionalPixelSeedGeneratorSingleTop)
openHltBLifetimeRegionalPixelSeedGenerator.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag(BJetinputjetCollection)

openHltBLifetimeRegionalCkfTrackCandidates = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidatesSingleTop)
openHltBLifetimeRegionalCkfTrackCandidates.src = cms.InputTag("openHltBLifetimeRegionalPixelSeedGenerator")

openHltBLifetimeRegionalCtfWithMaterialTracks = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracksSingleTop)
openHltBLifetimeRegionalCtfWithMaterialTracks.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidates")

openHltBLifetimeL3Associator = copy.deepcopy(hltBLifetimeL3AssociatorSingleTop)
openHltBLifetimeL3Associator.jets   = cms.InputTag(BJetinputjetCollection)
openHltBLifetimeL3Associator.tracks = cms.InputTag("openHltBLifetimeRegionalCtfWithMaterialTracks")

openHltBLifetimeL3TagInfos = copy.deepcopy(hltBLifetimeL3TagInfosSingleTop)
openHltBLifetimeL3TagInfos.jetTracks = cms.InputTag("openHltBLifetimeL3Associator")

openHltBLifetimeL3BJetTags = copy.deepcopy(hltBLifetimeL3BJetTagsSingleTop)
openHltBLifetimeL3BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfos"))

# Single Track TC
openHltBLifetimeL3BJetTagsSingleTrack = copy.deepcopy(hltBLifetimeL3BJetTagsSingleTrack)
openHltBLifetimeL3BJetTagsSingleTrack.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfos"))

# L3 reco sequence for lifetime tagger
OpenHLTBLifetimeL3recoSequence = cms.Sequence(
    HLTDoLocalPixelSequence +
    HLTDoLocalStripSequence +
    openHltBLifetimeRegionalPixelSeedGenerator +
    openHltBLifetimeRegionalCkfTrackCandidates +
    openHltBLifetimeRegionalCtfWithMaterialTracks +
    openHltBLifetimeL3Associator +
    openHltBLifetimeL3TagInfos +
    openHltBLifetimeL3BJetTagsSingleTrack +
    openHltBLifetimeL3BJetTags )

### soft-muon-based b-tag OpenHLT (ideal, start up, and performance meas.) ####
# L1 filter is skipped
# L2 reco sequence is common to all paths, and taken from the global table
# L2.5 and L3 sequences are rewritten to bypass selectors and filters

openHltBSoftmuonL25TagInfos = copy.deepcopy(hltBSoftMuonDiJet20L25TagInfos)
openHltBSoftmuonL25TagInfos.jets = cms.InputTag(BJetinputjetCollection)

#### BTagMu paths make use of SoftMuonByDR both at L2.5 and L3
openHltBSoftmuonL25BJetTags = copy.deepcopy(hltBSoftMuonDiJet20L25BJetTagsByDR)
openHltBSoftmuonL25BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL25TagInfos"))

openHltBSoftmuonL3TagInfos = copy.deepcopy(hltBSoftMuonDiJet20Mu5SelL3TagInfos)
openHltBSoftmuonL3TagInfos.jets = cms.InputTag(BJetinputjetCollection)
openHltBSoftmuonL3TagInfos.leptons = cms.InputTag("hltL3Muons")  #Feed the entire L3Muons not the filtered ones

#### BTagMu paths make use of SoftMuonByDR both at L2.5 and L3
openHltBPerfMeasL3BJetTags = copy.deepcopy(hltBSoftMuonDiJet20Mu5SelL3BJetTagsByDR)
openHltBPerfMeasL3BJetTags.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL3TagInfos"))

OpenHLTBSoftMuonL25recoSequence = cms.Sequence(
    HLTL2muonrecoNocandSequence +
    openHltBSoftmuonL25TagInfos +
    openHltBSoftmuonL25BJetTags )

OpenHLTBSoftMuonL3recoSequence = cms.Sequence(
    HLTL3muonrecoNocandSequence +
    openHltBSoftmuonL3TagInfos +
    openHltBPerfMeasL3BJetTags )
