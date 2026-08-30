import FWCore.ParameterSet.Config as cms

# ---------------------------------------------------------------------------
# Replacement for:
#
#   hltL1SeedsForQuadPuppiJetTripleBtagFilter = cms.EDFilter("PathStatusFilter",
#       logicalExpression = cms.string('pPuppiHT400 and pQuadJet70_55_40_40')
#   )
#
# The original gates on two independent L1 paths firing simultaneously:
#
#   PuppiHT400        -> l1tGTSingleObjectCond on CL2HtSum  >= 400 GeV
#   QuadJet70554040   -> l1tGTQuadObjectCond   on CL2JetsSC4 (70, 55, 40, 40 GeV)
#
# Both conditions must fire.  We implement this as two sequential HLT
# filters on the same cms.Path: the first rejects events where PuppiHT400
# did not fire, the second rejects events where QuadJet70_55_40_40 did not
# fire.  The logical AND is enforced by the path short-circuit.
#
# The two filters also register the matched L1 objects in the trigger
# event so downstream filters (e.g. the b-tag filter) can retrieve refs.
# ---------------------------------------------------------------------------

# --- helper for open pair cuts (no inter-jet requirement in this seed) ----
_noPairCuts = cms.PSet(
    minDR      = cms.double(0.),
    maxDR      = cms.double(1e9),
    minDEta    = cms.double(-1.),   # disabled
    minDPhi    = cms.double(-1.),   # disabled
    minInvMass = cms.double(0.),
    maxInvMass = cms.double(1e9),
)

# --- 1. PuppiHT400 seed ---------------------------------------------------
#
# l1tGTSingleObjectCond on CL2HtSum with scalar-sum pT >= 400 GeV.
# CL2HtSum is a single-object collection (one entry per event), so minN=1
# and the pT threshold on the single entry is sufficient.

hltL1PuppiHT400 = cms.EDFilter("HLTP2GTSingleObjectFilter",
    saveTags         = cms.bool(True),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    minN             = cms.uint32(1),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name       = cms.string("pPuppiHT400_pQuadJet70_55_40_40"),
            collection = cms.PSet(
                objectType = cms.string("CL2HtSum"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(1e9),   # HT sums have no eta
            ),
        ),
    ),
)

# --- 2. QuadJet70_55_40_40 seed ------------------------------------------
#
# l1tGTQuadObjectCond on CL2JetsSC4.
# All four legs draw from the same product (same InputTag / same ProductID),
# so the deduplication in HLTP2GTQuadObjectFilter will enforce index ordering
# between legs with the same threshold (legs 3 and 4, both at 40 GeV),
# while still testing all role assignments between legs with different
# thresholds (legs 1, 2, 3/4).
#
# No inter-jet cuts in the original L1 condition -> all pair-cut PSets open.

hltL1QuadJet70554040 = cms.EDFilter("HLTP2GTQuadObjectFilter",
    saveTags         = cms.bool(True),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name = cms.string("pPuppiHT400_pQuadJet70_55_40_40"),
            collection1 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection2 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection3 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            collection4 = cms.PSet(
                objectType = cms.string("CL2JetsSC4"),
                minPt      = cms.double(0.),
                maxAbsEta  = cms.double(99.),
            ),
            cuts12 = _noPairCuts,
            cuts13 = _noPairCuts,
            cuts14 = _noPairCuts,
            cuts23 = _noPairCuts,
            cuts24 = _noPairCuts,
            cuts34 = _noPairCuts,
        ),
    ),
)

# --- Combined seed (replaces hltL1SeedsForQuadPuppiJetTripleBtagFilter) ---
#
# Use this cms.Sequence at the start of the HLT path that formerly used
# hltL1SeedsForQuadPuppiJetTripleBtagFilter.  The path short-circuit
# provides the logical AND.
#
#   Before:
#     hltL1SeedsForQuadPuppiJetTripleBtagFilter  (PathStatusFilter)
#
#   After:
#     hltL1PuppiHT400 + hltL1QuadJet70554040     (as a cms.Sequence)

hltL1SeedsForQuadPuppiJetTripleBtagFilter = cms.Sequence(
    hltL1PuppiHT400 +
    hltL1QuadJet70554040
)
