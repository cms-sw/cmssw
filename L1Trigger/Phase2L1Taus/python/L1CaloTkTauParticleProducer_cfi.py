
import FWCore.ParameterSet.Config as cms

L1CaloTkTaus = cms.EDProducer("L1CaloTkTauParticleProducer",
        label = cms.string("CaloTk"), # labels the collection of L1CaloTkTauParticleProducer that is produced

        # L1 tracks
     	L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
        tk_nFitParams   = cms.uint32( 4 ),
        tk_minPt        = cms.double( 2.0 ),
        tk_minEta       = cms.double( 0.0 ),
        tk_maxEta       = cms.double( 2.5 ),
        tk_maxChiSq     = cms.double( 94.0 ),
        tk_minStubs     = cms.uint32( 5 ),

        # Seed tracks
        seedTk_minPt        = cms.double( 5.0 ), 
        seedTk_minEta       = cms.double( 0.0 ),
        seedTk_maxEta       = cms.double( 2.5 ),
        seedTk_maxChiSq     = cms.double( 94.0 ),
        seedTk_minStubs     = cms.uint32( 5 ),
        seedTk_useMaxDeltaR = cms.bool( False ), 
        seedTk_maxDeltaR    = cms.double( 0.15 ),

        # Matching parameters
        matching_maxDeltaR  = cms.double( 0.1 ),
        matchHighestPt      = cms.bool( False ),         

        # Signal cone and clustering parameters
        shrinkCone_Constant   = cms.double( 5.0 ), #TkEGs: 2.5
        sigCone_cutoffDeltaR  = cms.double( 0.25), #TkEGs: 0.15
        sigCone_dRMin         = cms.double( 0.0 ), 
        sigConeTks_maxDeltaZ  = cms.double( 1.00 ), # cm
        sigConeTks_maxInvMass = cms.double( 1.50 ), # GeV 

        # Isolation cone parameters
        isoCone_useCone   = cms.bool( True ), # instead of annulus
        isoCone_dRMax     = cms.double( 0.40 ),

        # CaloTau parameters
        L1CaloTauInputTag   = cms.untracked.VInputTag("simCaloStage2Digis"), #same as in L1Trigger/L1TNtuples/python/l1PhaseIITreeProducer_cfi.py
        caloTauEtMin        = cms.double( 0.0 ),
        calibrateCaloTaus   = cms.bool( False ),

        # Isolation parameters
        useVtxIso = cms.bool( True ),
        vtxIso_WP = cms.double( 0.5 ), #TkEGs: 1.0
        relIso_maxDeltaZ = cms.double( 0.5 ),       
)
