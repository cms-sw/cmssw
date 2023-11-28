import FWCore.ParameterSet.Config as cms

hltHpsSelectedPFTausTrackFinding = cms.EDFilter( "PFTauSelector",
    src = cms.InputTag( "hltHpsPFTauProducer" ),
    cut = cms.string( "pt > 0" ),
    discriminators = cms.VPSet( 
      cms.PSet(  discriminator = cms.InputTag( "hltHpsPFTauTrackFindingDiscriminator" ),
        selectionCut = cms.double( 0.5 )
      )
    ),
    discriminatorContainers = cms.VPSet( 
    )
)
