# This file defines 5 "sinple" OpenHLT paths
#   OpenHLT_BTagIP
#   OpenHLT_BTagIP_Relaxed
#   OpenHLT_BTagMu
#   OpenHLT_BTagMu_Relaxed
#   OpenHLT_BTagPerformance
#
# and a "cumulative" path
#   DoHLTBTag
#
# which runs all the sequences from all the simple ones above, useing at L1 an OR of all their L1 filters 
#  (I arbitrarily lowered L1_Mu5_Jet15 to L1_Mu3_Jet15)


### Open-HLT includes, modules and paths for b-jet
import copy
import FWCore.ParameterSet.Config as cms

# import the whole HLT menu
from FastSimulation.Configuration.HLT_cff import *
  
### common sequences for b-tag ################################################
HLTBCommonL2recoSequence = cms.Sequence( 
    HLTDoCaloSequence + 
    HLTDoJetRecoSequence + 
    HLTDoHTRecoSequence )

### lifetime-based b-tag OpenHLT (ideal) ######################################
# L1, L2 and L2.5 sequences are taken directly from the global table
# L3 is rewritten to bypass the L2.5 filter

openHltBLifetimeRegionalPixelSeedGenerator = copy.deepcopy(hltBLifetimeRegionalPixelSeedGenerator)
#openHltBLifetimeRegionalPixelSeedGenerator.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("hltBLifetimeL25Jets")

openHltBLifetimeRegionalCkfTrackCandidates = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidates)
openHltBLifetimeRegionalCkfTrackCandidates.SeedProducer = cms.string("openHltBLifetimeRegionalPixelSeedGenerator")

openHltBLifetimeRegionalCtfWithMaterialTracks = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracks)
openHltBLifetimeRegionalCtfWithMaterialTracks.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidates")
  
openHltBLifetimeL3Associator = copy.deepcopy(hltBLifetimeL3Associator)
openHltBLifetimeL3Associator.jets   = cms.InputTag("hltBLifetimeL25Jets")
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

# OpenHLT path for lifetime tagger
#OpenHLT_BTagIP = cms.Path( 
#    HLTBeginSequence + 
#    hltBLifetimeL1seeds + 
#    HLTBCommonL2recoSequence + 
#    HLTBLifetimeL25recoSequence + 
#    OpenHLTBLifetimeL3recoSequence + 
#    HLTEndSequence )


### lifetime-based b-tag OpenHLT (start up) ###################################
# L1, L2 and L2.5 sequences are taken directly from the global table
# L3 is rewritten to bypass the L2.5 filter

openHltBLifetimeRegionalPixelSeedGeneratorRelaxed = copy.deepcopy(hltBLifetimeRegionalPixelSeedGeneratorRelaxed)
#openHltBLifetimeRegionalPixelSeedGeneratorRelaxed.RegionFactoryPSet.RegionPSet.JetSrc = cms.InputTag("hltBLifetimeL25JetsRelaxed")

openHltBLifetimeRegionalCkfTrackCandidatesRelaxed = copy.deepcopy(hltBLifetimeRegionalCkfTrackCandidatesRelaxed)
openHltBLifetimeRegionalCkfTrackCandidatesRelaxed.SeedProducer = cms.string("openHltBLifetimeRegionalPixelSeedGeneratorRelaxed")

openHltBLifetimeRegionalCtfWithMaterialTracksRelaxed = copy.deepcopy(hltBLifetimeRegionalCtfWithMaterialTracksRelaxed)
openHltBLifetimeRegionalCtfWithMaterialTracksRelaxed.src = cms.InputTag("openHltBLifetimeRegionalCkfTrackCandidatesRelaxed")
  
openHltBLifetimeL3AssociatorRelaxed = copy.deepcopy(hltBLifetimeL3AssociatorRelaxed)
openHltBLifetimeL3AssociatorRelaxed.jets   = cms.InputTag("hltBLifetimeL25JetsRelaxed")
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

# OpenHLT path for relaxed lifetime tagger
#OpenHLT_BTagIP_Relaxed = cms.Path( 
#    HLTBeginSequence + 
#    hltBLifetimeL1seedsLowEnergy + 
#    HLTBCommonL2recoSequence + 
#    HLTBLifetimeL25recoSequenceRelaxed + 
#    OpenHLTBLifetimeL3recoSequenceRelaxed + 
#    HLTEndSequence )


### soft-muon-based b-tag OpenHLT (ideal) #####################################
# L1 is rewritten to accept both n-jets and HTT events
# L2, L2.5 and L3 sequences are taken directly from the global table

openHltBSoftmuonL1seeds = copy.deepcopy(hltBSoftmuonNjetL1seeds)
openHltBSoftmuonL1seeds.L1SeedsLogicalExpression = cms.string("L1_Mu5_Jet15 OR L1_HTT300")

#OpenHLT_BTagMu  = cms.Path( 
#    HLTBeginSequence + 
#    openHltBSoftmuonL1seeds + 
#    HLTBCommonL2recoSequence + 
#    HLTBSoftmuonL25recoSequence + 
#    HLTBSoftmuonL3recoSequence + 
#    HLTEndSequence )


### soft-muon-based b-tag OpenHLT (start up) ##################################
# L1 is rewritten to accept both n-jets and HTT events
# L2, L2.5 and L3 sequences are taken directly from the global table

openHltBSoftmuonL1seedsLowEnergy = copy.deepcopy(hltBSoftmuonNjetL1seeds)
openHltBSoftmuonL1seedsLowEnergy.L1SeedsLogicalExpression = cms.string("L1_Mu5_Jet15 OR L1_HTT200")

#OpenHLT_BTagMu_Relaxed = cms.Path( 
#    HLTBeginSequence + 
#    openHltBSoftmuonL1seedsLowEnergy + 
#    HLTBCommonL2recoSequence + 
#    HLTBSoftmuonL25recoSequence + 
#    HLTBSoftmuonL3recoSequence + 
#    HLTEndSequence )


### b-tag performance measurement OpenHLT #####################################
# taken from the global table, removing filters

#OpenHLT_BTagPerformance = cms.Path( 
#    HLTBeginSequence + 
#    hltBSoftmuonNjetL1seeds + 
#    HLTBCommonL2recoSequence + 
#    HLTBSoftmuonL25recoSequence + 
#    HLTBSoftmuonL3recoSequence + 
#    HLTEndSequence )


### L1 seeds module to run all b-tag OpenHLT in a single path (ideal) #########
# L1 is an OR of all L1 filters

openHltBL1seeds = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding     = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet150 OR L1_DoubleJet100 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT300 OR L1_Mu5_Jet15" ),
    L1GtReadoutRecordTag     = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag         = cms.InputTag( "gtDigis" ),
    L1CollectionsTag         = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag      = cms.InputTag( "l1ParamMuons" )
)


### L1 seeds module to run all b-tag OpenHLT in a single path (start up) ######
# L1 is an OR of all L1 filters (L1_Mu5_Jet15 arbitrarily lowered to L1_Mu3_Jet15)

openHltBL1seedsLowEnergy = cms.EDFilter( "HLTLevel1GTSeed",
    L1TechTriggerSeeding     = cms.bool( False ),
    L1SeedsLogicalExpression = cms.string( "L1_SingleJet100 OR L1_DoubleJet70 OR L1_TripleJet50 OR L1_QuadJet30 OR L1_HTT200 OR L1_Mu3_Jet15" ),
    L1GtReadoutRecordTag     = cms.InputTag( "gtDigis" ),
    L1GtObjectMapTag         = cms.InputTag( "gtDigis" ),
    L1CollectionsTag         = cms.InputTag( "l1extraParticles" ),
    L1MuonCollectionTag      = cms.InputTag( "l1ParamMuons" )
)


