import FWCore.ParameterSet.Config as cms

L1TkElectrons = cms.EDProducer("L1TkElectronTrackProducer",
	  label = cms.string("EG"),	# labels the collection of L1TkEmParticleProducer that is produced.
		# (not really needed actually)
    L1EGammaInputTag = cms.InputTag("simCaloStage2Digis",""),
    # Only the L1EG objects that have ET > ETmin in GeV
    # are considered. ETmin < 0 means that no cut is applied.
    ETmin = cms.double( -1.0 ),
    L1TrackInputTag = cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks"),
    # Quality cuts on Track and Track L1EG matching criteria
    TrackChi2           = cms.double(1e10), # minimum Chi2 to select tracks
    TrackMinPt          = cms.double(10.0), # minimum Pt to select tracks
    useTwoStubsPT       = cms.bool( False ),
    useClusterET       = cms.bool( False ),
    TrackEGammaMatchType = cms.string("PtDependentCut"),
    TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0), # functional Delta Phi cut parameters to match Track with L1EG objects
    TrackEGammaDeltaR   = cms.vdouble(0.08, 0.0, 0.0), # functional Delta R cut parameters to match Track with L1EG objects
    TrackEGammaDeltaEta = cms.vdouble(1e10, 0.0, 0.0), # Delta Eta cutoff to match Track with L1EG objects
                                                       # are considered. (unused in default configuration)
    RelativeIsolation = cms.bool( True ),	# default = True. The isolation variable is relative if True,
						# else absolute.
    # Cut on the (Trk-based) isolation: only the L1TkEmParticle for which
    # the isolation is below RelIsoCut are written into
    # the output collection. When RelIsoCut < 0, no cut is applied.
		# When RelativeIsolation = False, IsoCut is in GeV.
    # Determination of the isolation w.r.t. L1Tracks :
    IsoCut = cms.double( -0.10 ),
    PTMINTRA = cms.double( 2. ),	# in GeV
	  DRmin = cms.double( 0.03),
	  DRmax = cms.double( 0.2 ),
    maxChi2IsoTracks = cms.double(1e10), # max chi2 cut for a track to be considered for isolation computation
    minNStubsIsoTracks = cms.int32(0), # min cut on # of stubs for a track to be considered for isolation computation
	  DeltaZ = cms.double( 0.6 )    # in cm. Used for tracks to be used isolation calculation
)
L1TkIsoElectrons = L1TkElectrons.clone(
    IsoCut = cms.double( 0.10 )
)
# for  LowPt Electron
L1TkElectronsLoose = L1TkElectrons.clone(
    TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0),
    TrackEGammaDeltaR = cms.vdouble(0.12, 0.0, 0.0),
    TrackMinPt = cms.double( 3.0 )
)


#### Additional collections that right now only the menu team is using - to be renamed/redefined by the EGamma group
# The important change is the EG seed -> PhaseII instead of PhaseI

#barrel
L1TkElectronsCrystal = L1TkElectrons.clone(
    L1EGammaInputTag = cms.InputTag("L1EGammaClusterEmuProducer"),
    IsoCut = cms.double(-0.1)
)

L1TkIsoElectronsCrystal=L1TkElectronsCrystal.clone(
    IsoCut = cms.double(0.1)
)

L1TkElectronsLooseCrystal = L1TkElectronsCrystal.clone(
    TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0),
    TrackEGammaDeltaR = cms.vdouble(0.12, 0.0, 0.0),
    TrackMinPt = cms.double( 3.0 )
)

L1TkElectronsEllipticMatchCrystal = L1TkElectronsCrystal.clone(
    TrackEGammaMatchType = cms.string("EllipticalCut"),
    TrackEGammaDeltaEta = cms.vdouble(0.015, 0.025,1e10)
)



#endcap
L1TkElectronsHGC=L1TkElectrons.clone(
    L1EGammaInputTag = cms.InputTag("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts"),
    IsoCut = cms.double(-0.1)
)


L1TkElectronsEllipticMatchHGC = L1TkElectronsHGC.clone(
    TrackEGammaMatchType = cms.string("EllipticalCut"),
    TrackEGammaDeltaEta = cms.vdouble(0.01, 0.01,1e10),
    maxChi2IsoTracks = cms.double(100),
    minNStubsIsoTracks = cms.int32(4),
)


L1TkIsoElectronsHGC=L1TkElectronsHGC.clone(
    DRmax = cms.double(0.4),
    DeltaZ = cms.double(1.0),
    maxChi2IsoTracks = cms.double(100),
    minNStubsIsoTracks = cms.int32(4),
    IsoCut = cms.double(0.1)
 )

L1TkElectronsLooseHGC = L1TkElectronsHGC.clone(
    TrackEGammaDeltaPhi = cms.vdouble(0.07, 0.0, 0.0),
    TrackEGammaDeltaR = cms.vdouble(0.12, 0.0, 0.0),
    TrackMinPt = cms.double( 3.0 )
)
