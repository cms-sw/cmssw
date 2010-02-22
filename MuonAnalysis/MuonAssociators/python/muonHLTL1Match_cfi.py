import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

muonHLTL1Match = cms.EDProducer("HLTL1MuonMatcher",
    src = cms.InputTag("muons"),

    # L1 Muon collection, and preselection on that collection
    matched     = cms.InputTag("patTrigger"),
    pathName    = cms.string('HLT_L1MuOpen'),
    filterLabel = cms.string(''), # set to a non-empty string to select L1 objects also on the filter label

    # Choice of matching algorithm
    useTrack = cms.string("tracker"),  # 'none' to use Candidate P4; or 'tracker', 'muon', 'global'
    useState = cms.string("atVertex"), # 'innermost' and 'outermost' require the TrackExtra (not AOD)
    useSimpleGeometry = cms.bool(True), # just use a cylinder plus two disks.

    # Matching Criteria
    maxDeltaR = cms.double(0.3),

    # Fake filter lavels for the object propagated to the second muon station
    setPropLabel = cms.string("propagatedToM2"),

    # Write extra ValueMaps
    writeExtraInfo = cms.bool(False),
)
