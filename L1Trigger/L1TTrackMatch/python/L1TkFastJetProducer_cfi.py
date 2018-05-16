import FWCore.ParameterSet.Config as cms

L1TkFastJets = cms.EDProducer("L1TkFastJetProducer",
     	L1TrackInputTag = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"),
	L1PrimaryVertexTag=cms.InputTag("L1TkPrimaryVertex","","L1TrackJets"), 
        TRK_ZMAX = cms.double(25.),         # max track z0 [cm]
        TRK_CHI2MAX = cms.double(100.),     # max track chi2 for all tracks
        TRK_PTMIN = cms.double(2.0),        # minimum track pt [GeV]
        TRK_ETAMAX = cms.double(2.5),       # maximum track eta
        TRK_NSTUBMIN = cms.int32(4),        # minimum number of stubs on track
        TRK_NSTUBPSMIN = cms.int32(2),      # minimum number of stubs in PS modules on track
	PVtxDeltaZ=cms.double(0.5),         #cluster tracks within |dz|<X
        doPtComp = cms.bool( False ),	   # track-stubs PT compatibility cut
    	doTightChi2 = cms.bool( True ),    # chi2dof < 5 for tracks with PT > 20
	L1Tk_nPar = cms.int32(4),	   # use 4 or 5-parameter L1 track fit
	CONESize=cms.double(0.4)           #cone size for anti-kt fast jet
	
)
