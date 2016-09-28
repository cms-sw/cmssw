import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from math import pi

muonL1MatcherParameters = cms.PSet(
    # Choice of matching algorithm
    useTrack = cms.string("tracker"),   # 'none' to use Candidate P4; or 'tracker', 'muon', 'global'
    useState = cms.string("atVertex"),  # 'innermost' and 'outermost' require the TrackExtra
    useSimpleGeometry = cms.bool(True),  # just use a cylinder plus two disks.
    fallbackToME1 = cms.bool(False),     # If propagation to ME2 fails, propagate to ME1

    useMB2InOverlap =  cms.bool(False),  # propagate to MB2 in overlap region (according to L1 experts OMTF uses MB2 as RF in all its coverage) 
    useStage2L1 = cms.bool(False),       # Use stage2 L1 instead of legacy one

    sortBy = cms.string("pt"),          # among compatible candidates, pick the highest pt one

    # Matching Criteria
    maxDeltaR   = cms.double(0.5),
    maxDeltaPhi = cms.double(6),
    maxDeltaEta = cms.double(99),
    l1PhiOffset = cms.double(1.25 * pi/180.), ## Offset to add to L1 phi before matching (according to L1 experts)
)

### For L1 Singlets you 

muonL1Match = cms.EDProducer("L1MuonMatcher",
    muonL1MatcherParameters,

    # Reconstructed muons
    src = cms.InputTag("muons"),

    # L1 Muon collection, and preselection on that collection
    matched      = cms.InputTag("l1extraParticles"),
    preselection = cms.string("bx == 0"),

    # Fake filter labels for output
    setL1Label = cms.string("l1"),
    setPropLabel = cms.string("propagated"),

    # Write extra ValueMaps
    writeExtraInfo = cms.bool(True),

    # Min and Max BXs from l1t::BxVector (applies to stage 2 only)
    firstBX = cms.int32(0),
    lastBX  = cms.int32(0),

)
