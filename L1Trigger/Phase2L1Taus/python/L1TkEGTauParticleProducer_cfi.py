
import FWCore.ParameterSet.Config as cms

L1TkEGTaus = cms.EDProducer("L1TkEGTauParticleProducer",
        label = cms.string("TkEG"), 	# labels the collection of L1TkEGTauParticleProducer that is produced

        # L1 Tracks
     	L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
        tk_nFitParams   = cms.uint32( 4 ),
        tk_minPt        = cms.double( 2.0 ),
        tk_minEta       = cms.double( 0.0 ),
        tk_maxEta       = cms.double( 1.5 ),
        tk_maxChiSq     = cms.double( 94.0 ),
        tk_minStubs     = cms.uint32( 5 ),
                            
        # L1 EGammas
        L1EGammaInputTag      = cms.InputTag("L1EGammaClusterEmuProducer", "L1EGammaCollectionBXVEmulator"), 
        #L1EGammaHGCalInputTag = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts"),
        eg_minEt              = cms.double( 1.5 ),
        eg_minEta             = cms.double( 0.0 ),
        eg_maxEta             = cms.double( 1.5 ),

        # Seed-tracks parameters
        seedtk_minPt     = cms.double( 5.0 ),
        seedtk_maxEta    = cms.double( 1.5 ),
        seedtk_maxChiSq  = cms.double( 94.0 ),
        seedtk_minStubs  = cms.uint32( 5 ),
        seedtk_maxDeltaR = cms.double( 0.15 ),

        # Shrinking Cone parameters
        shrinkCone_Constant  = cms.double( 2.5 ),
        sigCone_dRMin        = cms.double( 0.0 ), 
        #sigCone_dRMax        = cms.double( 0.15), 
        sigCone_cutoffDeltaR = cms.double( 0.15 ),
        isoCone_dRMax        = cms.double( 0.30 ),
        isoCone_useCone      = cms.bool( False ), # instead of annulus

        # Tracks & EGs clustering parameters
        maxDeltaZ_trks   = cms.double( 1.00 ), # cm
        maxInvMass_trks  = cms.double( 1.50 ), # GeV 
        maxInvMass_TkEGs = cms.double( 1.77 ), # GeV

        # Isolation parameters
        useVtxIso = cms.bool( True ),
        vtxIso_WP = cms.double( 1.0 ),
        
)
