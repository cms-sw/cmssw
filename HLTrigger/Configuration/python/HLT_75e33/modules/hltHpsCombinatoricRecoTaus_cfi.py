import FWCore.ParameterSet.Config as cms

hltHpsCombinatoricRecoTaus = cms.EDProducer( "RecoTauProducer",
    piZeroSrc = cms.InputTag( "hltPFTauPiZeros" ),
    jetRegionSrc = cms.InputTag( "hltTauPFJets08Region" ),
    maxJetAbsEta = cms.double( 2.5 ),
    outputSelection = cms.string( "leadPFChargedHadrCand().isNonnull()" ),
    chargedHadronSrc = cms.InputTag( "hltHpsTauPFJetsRecoTauChargedHadronsWithNeutrals" ),
    minJetPt = cms.double( 14.0 ),
    jetSrc = cms.InputTag( "hltAK4PFJets" ),
    builders = cms.VPSet( 
        cms.PSet(  decayModes = cms.VPSet( 
            cms.PSet(  maxPiZeros = cms.uint32( 0 ),
                       maxTracks = cms.uint32( 6 ),
                       nPiZeros = cms.uint32( 0 ),
                       nCharged = cms.uint32( 1 )
                   ),
            cms.PSet(  maxPiZeros = cms.uint32( 6 ),
                       maxTracks = cms.uint32( 6 ),
                       nCharged = cms.uint32( 1 ),
                       nPiZeros = cms.uint32( 1 )
                   ),
            cms.PSet(  maxPiZeros = cms.uint32( 5 ),
                       maxTracks = cms.uint32( 6 ),
                       nCharged = cms.uint32( 1 ),
                       nPiZeros = cms.uint32( 2 )
                   ),
            cms.PSet(  maxPiZeros = cms.uint32( 0 ),
                       maxTracks = cms.uint32( 6 ),
                       nCharged = cms.uint32( 2 ),
                       nPiZeros = cms.uint32( 0 )
                   ),
            cms.PSet(  maxPiZeros = cms.uint32( 3 ),
                       maxTracks = cms.uint32( 6 ),
                       nCharged = cms.uint32( 2 ),
                       nPiZeros = cms.uint32( 1 )
                   ),
            cms.PSet(  maxPiZeros = cms.uint32( 0 ),
                       maxTracks = cms.uint32( 6 ),
                       nCharged = cms.uint32( 3 ),
                       nPiZeros = cms.uint32( 0 )
                   ),
            cms.PSet(  maxPiZeros = cms.uint32( 3 ),
                       maxTracks = cms.uint32( 6 ),
                       nCharged = cms.uint32( 3 ),
                       nPiZeros = cms.uint32( 1 )
                   )
        ),
        isolationConeSize = cms.double( 0.5 ),
        minAbsPhotonSumPt_insideSignalCone = cms.double( 2.5 ),
        minAbsPhotonSumPt_outsideSignalCone = cms.double( 1.0E9 ),
        minRelPhotonSumPt_insideSignalCone = cms.double( 0.1 ),
        minRelPhotonSumPt_outsideSignalCone = cms.double( 1.0E9 ),
        name = cms.string( "combinatoric" ),
        pfCandSrc = cms.InputTag( "particleFlowTmp" ),
        plugin = cms.string( "RecoTauBuilderCombinatoricPlugin" ),
        qualityCuts = cms.PSet( 
            isolationQualityCuts = cms.PSet( 
                maxDeltaZ = cms.double( 0.2 ),
                maxTrackChi2 = cms.double( 100.0 ),
                maxTransverseImpactParameter = cms.double( 0.03 ),
                minGammaEt = cms.double( 1.5 ),
                minTrackHits = cms.uint32( 3 ),
                minTrackPixelHits = cms.uint32( 0 ),
                minTrackPt = cms.double( 1.0 ),
                minTrackVertexWeight = cms.double( -1.0 )
            ),
            leadingTrkOrPFCandOption = cms.string( "leadPFCand" ),
            primaryVertexSrc = cms.InputTag( "hltPhase2PixelVertices" ),
            pvFindingAlgo = cms.string( "closestInDeltaZ" ),
            recoverLeadingTrk = cms.bool( False ),
            signalQualityCuts = cms.PSet( 
                maxDeltaZ = cms.double( 0.4 ),
                maxTrackChi2 = cms.double( 1000.0 ),
                maxTransverseImpactParameter = cms.double( 0.2 ),
                minGammaEt = cms.double( 0.5 ),
                minNeutralHadronEt = cms.double( 30.0 ),
                minTrackHits = cms.uint32( 3 ),
                minTrackPixelHits = cms.uint32( 0 ),
                minTrackPt = cms.double( 0.5 ),
                minTrackVertexWeight = cms.double( -1.0 )
            ),
            vertexTrackFiltering = cms.bool( False ),
            vxAssocQualityCuts = cms.PSet( 
                maxTrackChi2 = cms.double( 1000.0 ),
                maxTransverseImpactParameter = cms.double( 0.2 ),
                minGammaEt = cms.double( 0.5 ),
                minTrackHits = cms.uint32( 3 ),
                minTrackPixelHits = cms.uint32( 0 ),
                minTrackPt = cms.double( 0.5 ),
                minTrackVertexWeight = cms.double( -1.0 )
            )
        ),
        signalConeSize = cms.string( "max(min(0.1, 3.0/pt()), 0.05)" )
        )
    ),
    buildNullTaus = cms.bool( False ),
    verbosity = cms.int32( 0 ),
    modifiers = cms.VPSet( 
        cms.PSet(  DataType = cms.string( "AOD" ),
                   EcalStripSumE_deltaEta = cms.double( 0.03 ),
                   EcalStripSumE_deltaPhiOverQ_maxValue = cms.double( 0.5 ),
                   EcalStripSumE_deltaPhiOverQ_minValue = cms.double( -0.1 ),
                   EcalStripSumE_minClusEnergy = cms.double( 0.1 ),
                   ElectronPreIDProducer = cms.InputTag( "elecpreid" ),
                   maximumForElectrionPreIDOutput = cms.double( -0.1 ),
                   name = cms.string( "elec_rej" ),
                   plugin = cms.string( "RecoTauElectronRejectionPlugin" ),
                   ElecPreIDLeadTkMatch_maxDR = cms.double( 0.01 )
               ),
        cms.PSet(  name = cms.string( "tau_mass" ),
                   plugin = cms.string( "PFRecoTauMassPlugin" ),
                   verbosity = cms.int32( 0 )
               )
    )
)
