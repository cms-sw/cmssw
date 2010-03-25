import FWCore.ParameterSet.Config as cms

particleFlowDisplacedVertex = cms.EDProducer("PFDisplacedVertexProducer",

    vertexCandidatesLabel = cms.InputTag("particleFlowDisplacedVertexCandidate"),

    # verbosity 
    verbose = cms.untracked.bool(True),
                                   
    # Debug flag
    debug = cms.untracked.bool(True),

    # maximum transverse distance between two points to be used in Seed                                   
    transvSize = cms.double(1.0),

    # maximum longitudinal distance between two points to be used in Seed                                   
    longSize = cms.double(5),

    # minimal radius below which we do not reconstruct interactions
    # typically the position of the first Pixel layer
    primaryVertexCut = cms.double(2.2),

    # radius below which we don't wamt to reconstruct displaced
    # vertices
    tobCut = cms.double(100),

    # z below which we don't want to reconstruct displaced
    # vertices
    tecCut = cms.double(220),

    # the minimal accepted weight for the tracks calculated in the 
    # adaptive vertex fitter to be associated to the displaced vertex
    # this correspond to the sigmacut of 6
    minAdaptWeight = cms.double(0.5),

                                             
    # ------------ Paramemeters for the track selection ------------

    # Primary vertex information used for dxy calculation
    mainVertexLabel = cms.InputTag("offlinePrimaryVertices", ""),
    offlineBeamSpotLabel = cms.InputTag("offlineBeamSpot", ""),

    # Parameters used to apply cuts
    tracksSelectorParameters = cms.PSet(
        bSelectTracks = cms.bool(True),
        # If a track is high purity it is always kept
        quality = cms.string("HighPurity"),
        # Following cuts are applyed to non high purity tracks
        # nChi2_max and pt_min cuts are applyed to the primary and secondary tracks
        nChi2_max = cms.double(5.),
        pt_min = cms.double(.2),
        # nChi2_min applyed only to primary tracks which may be short
        # remove fake pixel triplets
        nChi2_min = cms.double(.5),
        # Cuts applyed to the secondary tracks long and displaced
        dxy_min = cms.double(.2),
        nHits_min = cms.int32(6),
        nOuterHits_max = cms.int32(9)
    ),

    # ------------ Paramemeters for the vertex identification ------------

    vertexIdentifierParameters = cms.PSet(
        bIdentifyVertices = cms.bool(True),
        # Minimal sum pt of secondary tracks for displaced vertices.
        # Below this value we find either loopers splitted in two parts eiter
        # fake vertices in forward direction
        pt_min = cms.double(.2),
        # Minimal pT and log10(P_primary/P_secondary) for primary track in kinks (Primary+Secondary)
        # which are not identifier as K-+ decays
        pt_kink_min = cms.double(1.4),
        logPrimSec_min = cms.double(0.2),
        # maximum absoluta value of eta for loopers
        looper_eta_max = cms.double(0.1),
        # Masses cuts for selections
        #                    CVmin  K0min  K0max  K-min  K-max  Ldmin  Ldmax  Nuclmin_ee
        masses = cms.vdouble(0.050, 0.470, 0.525, 0.470, 0.525, 1.107, 1.125, 0.200),
        # Angle between the primaryVertex-secondaryVertex direction and secondary tracks direction
        # this angle means that the final system shall propagate in the same direction than initial system
        #                    all_max, CV and V0 max
        angles = cms.vdouble(60,      40)
    ),

    # Adaptive Vertex Fitter parameters identical to the default ones except sigmacut.
    # The default value is sigmacut = 3 too tight for displaced vertices
    # see CMS NOTE-2008/033 for more details
    avfParameters = cms.PSet(
        sigmacut = cms.double(6.),
        Tini = cms.double(256.),
        ratio = cms.double(0.25)
    )


)


