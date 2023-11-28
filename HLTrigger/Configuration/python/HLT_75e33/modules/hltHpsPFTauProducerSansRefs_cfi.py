import FWCore.ParameterSet.Config as cms

hltHpsPFTauProducerSansRefs = cms.EDProducer( "RecoTauCleaner",
    outputSelection = cms.string( "" ),
    cleaners = cms.VPSet( 
      cms.PSet(  name = cms.string( "HPS_Select" ),
        plugin = cms.string( "RecoTauDiscriminantCleanerPlugin" ),
        src = cms.InputTag( "hltHpsSelectionDiscriminator" )
      ),
      cms.PSet(  name = cms.string( "killSoftTwoProngTaus" ),
        plugin = cms.string( "RecoTauSoftTwoProngTausCleanerPlugin" ),
        minTrackPt = cms.double( 5.0 )
      ),
      cms.PSet(  name = cms.string( "ChargedHadronMultiplicity" ),
        plugin = cms.string( "RecoTauChargedHadronMultiplicityCleanerPlugin" )
      ),
      cms.PSet(  name = cms.string( "Pt" ),
        plugin = cms.string( "RecoTauStringCleanerPlugin" ),
        selection = cms.string( "leadPFCand().isNonnull()" ),
        selectionFailValue = cms.double( 1000.0 ),
        selectionPassFunction = cms.string( "-pt()" ),
        tolerance = cms.double( 0.01 )
      ),
      cms.PSet(  name = cms.string( "StripMultiplicity" ),
        plugin = cms.string( "RecoTauStringCleanerPlugin" ),
        selection = cms.string( "leadPFCand().isNonnull()" ),
        selectionFailValue = cms.double( 1000.0 ),
        selectionPassFunction = cms.string( "-signalPiZeroCandidates().size()" )
      ),
      cms.PSet(  name = cms.string( "CombinedIsolation" ),
        plugin = cms.string( "RecoTauStringCleanerPlugin" ),
        selection = cms.string( "leadPFCand().isNonnull()" ),
        selectionFailValue = cms.double( 1000.0 ),
        selectionPassFunction = cms.string( "isolationPFChargedHadrCandsPtSum() + isolationPFGammaCandsEtSum()" )
      )
    ),
    verbosity = cms.int32( 0 ),
    src = cms.InputTag( "hltHpsCombinatoricRecoTaus" )
)
