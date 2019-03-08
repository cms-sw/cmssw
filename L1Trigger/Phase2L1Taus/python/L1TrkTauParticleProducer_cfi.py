
import FWCore.ParameterSet.Config as cms

L1TrkTaus = cms.EDProducer("L1TrkTauParticleProducer",
        label = cms.string("TrkTau"), 	# labels the collection of L1TrkTauParticleProducer that is produced

        # L1 Tracks
     	L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
        tk_nFitParams   = cms.uint32( 4 ),
        tk_minPt        = cms.double( 2.0 ),
        tk_minEta       = cms.double( 0.0 ),
        tk_maxEta       = cms.double( 2.5 ),
        tk_maxChiSq     = cms.double( 94.0 ),
        tk_useRedChiSq  = cms.bool( False ),
        tk_minStubs     = cms.uint32( 5 ),
                            
        # Seed-tracks parameters
        seedtk_minPt       = cms.double( 5.0 ),
        seedtk_maxEta      = cms.double( 2.5 ),
        seedtk_maxChiSq    = cms.double( 94.0 ),
        seedtk_useRedChiSq = cms.bool( False ),
        seedtk_minStubs    = cms.uint32( 5 ),
        seedtk_maxDeltaR   = cms.double( 0.15 ),

        # Shrinking Cone parameters
        shrinkCone_Constant  = cms.double( 2.5 ),
        sigCone_dRMin        = cms.double( 0.0 ), 
        # sigCone_dRMax        = cms.double( 0.15), # not needed in shrinking cone mode
        sigCone_cutoffDeltaR = cms.double( 0.15 ),
        isoCone_dRMax        = cms.double( 0.30 ),
        isoCone_useCone      = cms.bool( False ),    # set to True (False) for cone (annulus)

        # Track-clustering parameters
        maxDeltaZ_trks  = cms.double( 1.00 ), # cm
        maxInvMass_trks = cms.double( 1.50 ), # GeV 

        # Isolation parameters
        useVtxIso = cms.bool( True ),
        vtxIso_WP = cms.double( 1.0 ),
        
)
