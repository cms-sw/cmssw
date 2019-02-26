
import FWCore.ParameterSet.Config as cms

L1CaloTkTaus = cms.EDProducer("L1CaloTkTauParticleProducer",
        label = cms.string("L1CaloTkTaus"), 	# labels the collection of L1CaloTkTauParticleProducer that is produced

#        # L1 Tracks
#     	L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
#        tk_nFitParams   = cms.uint32( 4 ),
#        tk_minPt        = cms.double( 2.0 ),
#        tk_minEta       = cms.double( 0.0 ),
#        tk_maxEta       = cms.double( 1.5 ),
#        tk_maxChiSq     = cms.double( 94.0 ),
#        tk_minStubs     = cms.uint32( 5 ),
#                            
#        # L1 EGammas
#        L1EGammaInputTag = cms.InputTag("L1EGammaClusterEmuProducer", "L1EGammaCollectionBXVEmulator"), 
#        eg_minEt         = cms.double( 1.5 ),
#        eg_minEta        = cms.double( 0.0 ),
#        eg_maxEta        = cms.double( 1.5 ),
#
#        # Seed-tracks parameters
#        seedtk_minPt     = cms.double( 5.0 ),
#        seedtk_maxEta    = cms.double( 1.5 ),
#        seedtk_maxChiSq  = cms.double( 94.0 ),
#        seedtk_minStubs  = cms.uint32( 5 ),
#        seedtk_maxDeltaR = cms.double( 0.15 ),
#
#        # Shrinking Cone parameters
#        shrinkCone_Constant  = cms.double( 2.5 ),
#        sigCone_dRMin        = cms.double( 0.0 ), 
#        #sigCone_dRMax        = cms.double( 0.15), 
#        sigCone_cutoffDeltaR = cms.double( 0.15 ),
#        isoCone_dRMax        = cms.double( 0.30 ),
#        isoCone_useCone      = cms.bool( False ), # instead of annulus
#
#        # Tracks & EGs clustering parameters
#        maxDeltaZ_trks  = cms.double( 1.00 ), # cm
#        maxInvMass_trks = cms.double( 1.50 ), # GeV 
#        maxInvMass_EGs  = cms.double( 1.77 ), # GeV
#
#        # Isolation parameters
#        useVtxIso = cms.bool( True ),
#        vtxIso_WP = cms.double( 0.50 ),
        
# NEW ONES

        # Common to all tracks
     	L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
        tk_nFitParams   = cms.uint32( 4 ),

        # Seed tracks
        seedTk_minPt    = cms.double( 5.0 ), 
        seedTk_minEta   = cms.double( 0.0 ),
        seedTk_maxEta   = cms.double( 2.5 ),
        seedTk_maxChiSq = cms.double( 94.0 ),
        seedTk_minStubs = cms.uint32( 5 ),

        # Signal / icolation cone tracks #FIXME: rename as "cone tracks", as the same sete is used for signal and isolation cones
        sigConeTks_minPt      = cms.double( 2.0 ),
        sigConeTks_minEta     = cms.double( 0.0 ),
        sigConeTks_maxEta     = cms.double( 2.5 ),
        sigConeTks_maxChiSq   = cms.double( 94.0 ),
        sigConeTks_minStubs   = cms.uint32( 5 ),
 
        # Signal cone parameters
        sigConeTks_dPOCAz     = cms.double( 1.0 ),
        sigConeTks_maxInvMass = cms.double( 1.5 ), 
        shrinkCone_Constant  = cms.double( 5.0 ), # in GeV, constant used for both signal and isolatio cones
        sigCone_dRMin        = cms.double( 0.0 ), # WARNING! If > 0 the matching Track will NOT be added in sigCone_TTTracks
        sigCone_dRMax        = cms.double( 0.25 ), # TODO: is this used for anything?
        sigCone_cutoffDeltaR = cms.double( 0.25), # this is used! = sigCone_dRMax

        # Isolation cone / annulus parameters
        isoCone_dRMin     = cms.double( 0.25 ), # = sigCone_dRMax
        isoCone_dRMax     = cms.double( 0.40 ),
        isoCone_useCone   = cms.bool( True ), # instead of annulus

        #CaloTaus
#        L1CaloTauInputTag   = cms.untracked.VInputTag(cms.InputTag("caloStage2Digis","Tau")), # used in L1Trigger/L1TNtuples/python/l1UpgradeTree_cfi.py
        L1CaloTauInputTag   = cms.untracked.VInputTag("simCaloStage2Digis"), #used in L1Trigger/L1TNtuples/python/l1PhaseIITreeProducer_cfi.py
#        L1CaloTauInputTag   = cms.InputTag("simCaloStage2Digis",""), # used in L1Trigger/L1TTrackMatch/python/L1TkTauFromCaloProducer_cfi.py
        calibrateCaloTaus   = cms.bool( True ),

        # Isolation criteria
        tau_jetWidth      = cms.double( 999.0 ), # one could try e.g. 0.50
        tau_vtxIsoWP      = cms.double( 0.50 ), 
        tau_relIsoWP      = cms.double( 0.15 ), 
        tau_relIsodZ0     = cms.double( 0.50 ),         
       
)
