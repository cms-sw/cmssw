import FWCore.ParameterSet.Config as cms

from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

muonHLTL1Match = cms.EDProducer("HLTL1MuonMatcher",
    src = cms.InputTag("muons"),

    # L1 Muon collection, and preselection on that collection
    matched     = cms.InputTag("patTrigger"),

    # Requests to select the object
    # 90% compatible with documentation at SWGuidePATTrigger#Module_Configuration_AN1
    andOr          = cms.bool( False ), # if False, do the 'AND' of the conditions below; otherwise, do the OR
    filterIdsEnum  = cms.vstring( '*' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_L1MuOpen' ),
    collectionTags = cms.vstring( '*' ),
    resolveAmbiguities    = cms.bool( True ), # if True, no more than one reco object can be matched to the same L1 object; precedence is given to the reco ones coming first in the list

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
