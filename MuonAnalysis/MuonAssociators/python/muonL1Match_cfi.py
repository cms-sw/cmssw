import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

from math import pi

muonL1Match = cms.EDProducer("L1MuonMatcher",
    src = cms.InputTag("muons"),

    # L1 Muon collection, and preselection on that collection
    matched      = cms.InputTag("l1extraParticles"),
    preselection = cms.string(""),

    # Choice of matching algorithm
    useTrack = cms.string("tracker"),  # 'none' to use Candidate P4; or 'tracker', 'muon', 'global'
    useState = cms.string("atVertex"), # 'innermost' and 'outermost' require the TrackExtra
    useSimpleGeometry = cms.bool(True), # just use a cylinder plus two disks.

    # Matching Criteria
    maxDeltaPhi = cms.double(6),
    maxDeltaR   = cms.double(0.3),
    l1PhiOffset = cms.double(1.25 * pi/180.), ## Offset to add to L1 phi before matching (according to L1 experts)

    # Fake filter lavels for output
    setL1Label = cms.string("l1"),
    setPropLabel = cms.string("propagated"),

    # Write extra ValueMaps
    writeExtraInfo = cms.bool(True),
)
