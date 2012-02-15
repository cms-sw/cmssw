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

###### Use corrected jets with L1FastJet PU subtraction
BJetinputjetCollectionL1FastJet="hltCaloJetL1FastJetCorrected"

openHltBLifetimeL25AssociatorL1FastJet = copy.deepcopy(hltBLifetimeL25AssociatorSingleTop)
openHltBLifetimeL25AssociatorL1FastJet.jets = cms.InputTag(BJetinputjetCollectionL1FastJet)

openHltBLifetimeL25TagInfosL1FastJet = copy.deepcopy(hltBLifetimeL25TagInfosSingleTop)
openHltBLifetimeL25TagInfosL1FastJet.jetTracks = cms.InputTag("openHltBLifetimeL25AssociatorL1FastJet")

openHltBLifetimeL25BJetTagsL1FastJet = copy.deepcopy(hltBLifetimeL25BJetTagsSingleTop)
openHltBLifetimeL25BJetTagsL1FastJet.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL25TagInfosL1FastJet"))


###### Use PFJets
BJetinputpfjetCollection="hltAntiKT5PFJets"


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

hltBLifetimeL25BJetTagsSingleTrackL1FastJet = cms.EDProducer( "JetTagProducer",
                                                     jetTagComputer = cms.string( "hltESPTrackCounting3D1st" ),
                                                     tagInfos = cms.VInputTag( 'openHltBLifetimeL25TagInfosL1FastJet' )
                                                     )


hltBLifetimeL3BJetTagsSingleTrackL1FastJet = cms.EDProducer( "JetTagProducer",
                                                    jetTagComputer = cms.string( "hltESPTrackCounting3D1st" ),
                                                    tagInfos = cms.VInputTag( 'openHltBLifetimeL3TagInfosL1FastJet' )
                                                    )

# Single Track TC
openHltBLifetimeL25BJetTagsSingleTrack = copy.deepcopy(hltBLifetimeL25BJetTagsSingleTrack)
openHltBLifetimeL25BJetTagsSingleTrack.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL25TagInfos"))

openHltBLifetimeL25BJetTagsSingleTrackL1FastJet = copy.deepcopy(hltBLifetimeL25BJetTagsSingleTrackL1FastJet)
openHltBLifetimeL25BJetTagsSingleTrackL1FastJet.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL25TagInfos"))




# L2.5 reco sequence for lifetime tagger
OpenHLTBLifetimeL25recoSequence = cms.Sequence(
        HLTDoLocalPixelSequence +
        HLTRecopixelvertexingSequence +
        openHltBLifetimeL25Associator +
        openHltBLifetimeL25TagInfos +
        openHltBLifetimeL25AssociatorL1FastJet +
        openHltBLifetimeL25TagInfosL1FastJet +
        openHltBLifetimeL25BJetTagsSingleTrack +
        openHltBLifetimeL25BJetTagsSingleTrackL1FastJet +
        openHltBLifetimeL25BJetTags +
        openHltBLifetimeL25BJetTagsL1FastJet )

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

# L1FastJet
openHltBLifetimeRegionalPixelSeedGeneratorL1FastJet = copy.deepcopy(hltBLifetimeRegionalPixelSeedGeneratorSingleTop)
openHltBLifetimeRegionalPixelSeedGeneratorL1FastJet.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag(BJetinputjetCollectionL1FastJet)

openHltBLifetimeRegionalCkfTrackCandidatesL1FastJet = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidatesSingleTop)
openHltBLifetimeRegionalCkfTrackCandidatesL1FastJet.src = cms.InputTag("openHltBLifetimeRegionalPixelSeedGeneratorL1FastJet")

openHltBLifetimeRegionalCtfWithMaterialTracksL1FastJet = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracksSingleTop)
openHltBLifetimeRegionalCtfWithMaterialTracksL1FastJet.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidatesL1FastJet")

openHltBLifetimeL3AssociatorL1FastJet = copy.deepcopy(hltBLifetimeL3AssociatorSingleTop)
openHltBLifetimeL3AssociatorL1FastJet.jets   = cms.InputTag(BJetinputjetCollectionL1FastJet)
openHltBLifetimeL3AssociatorL1FastJet.tracks = cms.InputTag("openHltBLifetimeRegionalCtfWithMaterialTracksL1FastJet")

openHltBLifetimeL3TagInfosL1FastJet = copy.deepcopy(hltBLifetimeL3TagInfosSingleTop)
openHltBLifetimeL3TagInfosL1FastJet.jetTracks = cms.InputTag("openHltBLifetimeL3AssociatorL1FastJet")

openHltBLifetimeL3BJetTagsL1FastJet = copy.deepcopy(hltBLifetimeL3BJetTagsSingleTop)
openHltBLifetimeL3BJetTagsL1FastJet.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfosL1FastJet"))

#PF jets
openHltBLifetimeL3AssociatorPF = copy.deepcopy(hltBLifetimeL3AssociatorSingleTop)
openHltBLifetimeL3AssociatorPF.jets   = cms.InputTag(BJetinputpfjetCollection)
openHltBLifetimeL3AssociatorPF.tracks = cms.InputTag("hltIter4Merged")
openHltBLifetimeL3AssociatorPF.pvSrc = cms.InputTag("hltPixelVertices")

openHltBLifetimeL3TagInfosPF = copy.deepcopy(hltBLifetimeL3TagInfosSingleTop)
openHltBLifetimeL3TagInfosPF.jetTracks = cms.InputTag("openHltBLifetimeL3AssociatorPF")

openHltBLifetimeL3BJetTagsPF = copy.deepcopy(hltBLifetimeL3BJetTagsSingleTop)
openHltBLifetimeL3BJetTagsPF.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfosPF"))

# Single Track TC
openHltBLifetimeL3BJetTagsSingleTrack = copy.deepcopy(hltBLifetimeL3BJetTagsSingleTrack)
openHltBLifetimeL3BJetTagsSingleTrack.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfos"))

openHltBLifetimeL3BJetTagsSingleTrackL1FastJet = copy.deepcopy(hltBLifetimeL3BJetTagsSingleTrackL1FastJet)
openHltBLifetimeL3BJetTagsSingleTrackL1FastJet.tagInfos = cms.VInputTag(cms.InputTag("openHltBLifetimeL3TagInfosL1FastJet"))

# L3 reco sequence for lifetime tagger
OpenHLTBLifetimeL3recoSequence = cms.Sequence(
    HLTDoLocalPixelSequence +
    HLTRecopixelvertexingSequence +
    HLTDoLocalStripSequence +
    HLTPFReconstructionSequence +
    openHltBLifetimeRegionalPixelSeedGenerator +
    openHltBLifetimeRegionalCkfTrackCandidates +
    openHltBLifetimeRegionalCtfWithMaterialTracks +
    openHltBLifetimeRegionalPixelSeedGeneratorL1FastJet +
    openHltBLifetimeRegionalCkfTrackCandidatesL1FastJet +
    openHltBLifetimeRegionalCtfWithMaterialTracksL1FastJet +
    openHltBLifetimeL3Associator +
    openHltBLifetimeL3TagInfos +
    openHltBLifetimeL3AssociatorL1FastJet +
    openHltBLifetimeL3TagInfosL1FastJet +
    openHltBLifetimeL3AssociatorPF +
    openHltBLifetimeL3TagInfosPF +
    openHltBLifetimeL3BJetTagsSingleTrack +
    openHltBLifetimeL3BJetTagsSingleTrackL1FastJet +
    openHltBLifetimeL3BJetTags +
    openHltBLifetimeL3BJetTagsL1FastJet +
    openHltBLifetimeL3BJetTagsPF )

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

# L1FastJet

openHltBSoftmuonL25TagInfosL1FastJet = copy.deepcopy(hltBSoftMuonDiJet20L25TagInfos)
openHltBSoftmuonL25TagInfosL1FastJet.jets = cms.InputTag(BJetinputjetCollectionL1FastJet)

#### BTagMu paths make use of SoftMuonByDR both at L2.5 and L3
openHltBSoftmuonL25BJetTagsL1FastJet = copy.deepcopy(hltBSoftMuonDiJet20L25BJetTagsByDR)
openHltBSoftmuonL25BJetTagsL1FastJet.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL25TagInfosL1FastJet"))

openHltBSoftmuonL3TagInfosL1FastJet = copy.deepcopy(hltBSoftMuonDiJet20Mu5SelL3TagInfos)
openHltBSoftmuonL3TagInfosL1FastJet.jets = cms.InputTag(BJetinputjetCollectionL1FastJet)
openHltBSoftmuonL3TagInfosL1FastJet.leptons = cms.InputTag("hltL3Muons")  #Feed the entire L3Muons not the filtered ones

#### BTagMu paths make use of SoftMuonByDR both at L2.5 and L3
openHltBPerfMeasL3BJetTagsL1FastJet = copy.deepcopy(hltBSoftMuonDiJet20Mu5SelL3BJetTagsByDR)
openHltBPerfMeasL3BJetTagsL1FastJet.tagInfos = cms.VInputTag(cms.InputTag("openHltBSoftmuonL3TagInfosL1FastJet"))



OpenHLTBSoftMuonL25recoSequence = cms.Sequence(
    HLTL2muonrecoNocandSequence +
    openHltBSoftmuonL25TagInfos +
    openHltBSoftmuonL25TagInfosL1FastJet +
    openHltBSoftmuonL25BJetTags +
    openHltBSoftmuonL25BJetTagsL1FastJet )

OpenHLTBSoftMuonL3recoSequence = cms.Sequence(
    HLTL3muonrecoNocandSequence +
    openHltBSoftmuonL3TagInfos +
    openHltBSoftmuonL3TagInfosL1FastJet +
    openHltBPerfMeasL3BJetTags +
    openHltBPerfMeasL3BJetTagsL1FastJet )
    
    
