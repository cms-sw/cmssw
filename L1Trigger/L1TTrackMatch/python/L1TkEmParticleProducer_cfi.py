import FWCore.ParameterSet.Config as cms
from L1Trigger.L1TTrackMatch.l1TkEmParticleProducer_cfi import l1TkEmParticleProducer

L1TkPhotons = l1TkEmParticleProducer.clone(
        label = "EG",                # labels the collection of L1TkEmParticleProducer that is produced
                                     # (not really needed actually)
        L1EGammaInputTag = ("simCaloStage2Digis",""),
                                     # When the standard sequences are used :
                                     #   - for the Run-1 algo, use ("l1extraParticles","NonIsolated")
                                     #     or ("l1extraParticles","Isolated")
                                     #   - for the "old stage-2" algo (2x2 clustering), use
                                     #     ("SLHCL1ExtraParticles","EGamma") or ("SLHCL1ExtraParticles","IsoEGamma")
                                     #   - for the new clustering algorithm of Jean-Baptiste et al,
                                     #     use ("SLHCL1ExtraParticlesNewClustering","IsoEGamma") or
                                     #     ("SLHCL1ExtraParticlesNewClustering","EGamma").
        ETmin = -1.,                 # Only the L1EG objects that have ET > ETmin in GeV
                                     # are considered. ETmin < 0 means that no cut is applied.
        RelativeIsolation = True,    # default = True. The isolation variable is relative if true, else absolute.
        IsoCut = 0.23,               # Cut on the (Trk-based) isolation: only the L1TkEmParticle for which
                                     # the isolation is below RelIsoCut are written into
                                     # the output collection. When RelIsoCut < 0, no cut is applied.
                                     # When RelativeIsolation = False, IsoCut is in GeV.
                                     # Determination of the isolation w.r.t. L1Tracks :
        L1TrackInputTag = ("TTTracksFromTrackletEmulation", "Level1TTTracks"),
        ZMAX = 25.,                   # in cm
        CHI2MAX = 100.,
        PTMINTRA = 2.,                # in GeV
        DRmin = 0.07,
        DRmax = 0.30,
        PrimaryVtxConstrain = False,  # default = False
                                      # if set to True, the default isolation is the PV constrained one, where L1TkPrimaryVertex
                                      #    is used to constrain the tracks entering in the calculation of the isolation
                                      # if set to False, the isolation is computed and stored, but not used 
        DeltaZMax =  0.6,             # in cm. Used only to compute the isolation with PrimaryVtxConstrain 
        L1VertexInputTag = ("L1TkPrimaryVertex")
)


L1TkPhotonsTightIsol = L1TkPhotons.clone(
    IsoCut = 0.10
)

#### Additional collections that right now only the menu team is using - to be renamed/redefined by the EGamma group
# The important change is the EG seed -> PhaseII instead of PhaseI

L1TkPhotonsCrystal=L1TkPhotons.clone(
    L1EGammaInputTag = ("L1EGammaClusterEmuProducer", ),
    IsoCut = -0.1
)

L1TkPhotonsHGC=L1TkPhotons.clone(
    L1EGammaInputTag = ("l1EGammaEEProducer","L1EGammaCollectionBXVWithCuts"),
    IsoCut = -0.1
)

