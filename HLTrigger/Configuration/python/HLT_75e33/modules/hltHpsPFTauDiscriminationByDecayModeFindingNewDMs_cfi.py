import FWCore.ParameterSet.Config as cms

hltHpsPFTauDiscriminationByDecayModeFindingNewDMs = cms.EDProducer( "PFRecoTauDiscriminationByHPSSelection",
    PFTauProducer = cms.InputTag( "hltHpsPFTauProducer" ),
    verbosity = cms.int32( 0 ),
    minTauPt = cms.double( 18.0 ),
    Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
    decayModes = cms.VPSet( 
      cms.PSet(  maxMass = cms.string( "1." ),
        nPiZeros = cms.uint32( 0 ),
        minMass = cms.double( -1000.0 ),
        nChargedPFCandsMin = cms.uint32( 1 ),
        nTracksMin = cms.uint32( 1 ),
        nCharged = cms.uint32( 1 ),
        applyBendCorrection = cms.PSet( 
          phi = cms.bool( True ),
          eta = cms.bool( True ),
          mass = cms.bool( True )
        )
      ),
      cms.PSet(  maxMass = cms.string( "max(1.72, min(1.72*sqrt(pt/100.), 4.2))" ),
        nPiZeros = cms.uint32( 1 ),
        minMass = cms.double( 0.0 ),
        nChargedPFCandsMin = cms.uint32( 1 ),
        nTracksMin = cms.uint32( 1 ),
        nCharged = cms.uint32( 1 ),
        assumeStripMass = cms.double( 0.1349 ),
        applyBendCorrection = cms.PSet( 
          phi = cms.bool( True ),
          eta = cms.bool( True ),
          mass = cms.bool( True )
        )
      ),
      cms.PSet(  minPi0Mass = cms.double( 0.0 ),
        maxMass = cms.string( "max(1.72, min(1.72*sqrt(pt/100.), 4.0))" ),
        maxPi0Mass = cms.double( 0.8 ),
        nPiZeros = cms.uint32( 2 ),
        minMass = cms.double( 0.4 ),
        nChargedPFCandsMin = cms.uint32( 1 ),
        nTracksMin = cms.uint32( 1 ),
        nCharged = cms.uint32( 1 ),
        assumeStripMass = cms.double( 0.0 ),
        applyBendCorrection = cms.PSet( 
          phi = cms.bool( True ),
          eta = cms.bool( True ),
          mass = cms.bool( True )
        )
      ),
      cms.PSet(  maxMass = cms.string( "1.2" ),
        nPiZeros = cms.uint32( 0 ),
        minMass = cms.double( 0.0 ),
        nChargedPFCandsMin = cms.uint32( 1 ),
        nTracksMin = cms.uint32( 2 ),
        nCharged = cms.uint32( 2 ),
        applyBendCorrection = cms.PSet( 
          phi = cms.bool( True ),
          eta = cms.bool( False ),
          mass = cms.bool( True )
        )
      ),
      cms.PSet(  maxMass = cms.string( "max(1.6, min(1.6*sqrt(pt/100.), 4.0))" ),
        minMass = cms.double( 0.0 ),
        nCharged = cms.uint32( 2 ),
        nChargedPFCandsMin = cms.uint32( 1 ),
        nPiZeros = cms.uint32( 1 ),
        nTracksMin = cms.uint32( 2 ),
        applyBendCorrection = cms.PSet( 
          eta = cms.bool( False ),
          phi = cms.bool( True ),
          mass = cms.bool( True )
        )
      ),
      cms.PSet(  maxMass = cms.string( "1.6" ),
        minMass = cms.double( 0.7 ),
        nCharged = cms.uint32( 3 ),
        nChargedPFCandsMin = cms.uint32( 1 ),
        nPiZeros = cms.uint32( 0 ),
        nTracksMin = cms.uint32( 2 ),
        applyBendCorrection = cms.PSet( 
          eta = cms.bool( False ),
          phi = cms.bool( True ),
          mass = cms.bool( True )
        )
      ),
      cms.PSet(  nCharged = cms.uint32( 3 ),
        nPiZeros = cms.uint32( 1 ),
        nTracksMin = cms.uint32( 2 ),
        minMass = cms.double( 0.9 ),
        maxMass = cms.string( "1.6" ),
        applyBendCorrection = cms.PSet( 
          eta = cms.bool( False ),
          phi = cms.bool( False ),
          mass = cms.bool( False )
        ),
        nChargedPFCandsMin = cms.uint32( 1 )
      )
    ),
    matchingCone = cms.double( 0.5 ),
    minPixelHits = cms.int32( 0 ),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool( False )
)
