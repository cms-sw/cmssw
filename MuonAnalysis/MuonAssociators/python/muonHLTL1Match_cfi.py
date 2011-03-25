import FWCore.ParameterSet.Config as cms

from MuonAnalysis.MuonAssociators.muonL1Match_cfi import *

muonHLTL1Match = cms.EDProducer("HLTL1MuonMatcher",
    muonL1MatcherParameters,

    # Reconstructed muons
    src = cms.InputTag("muons"),

    # L1 Muon collection, and preselection on that collection
    matched     = cms.InputTag("patTrigger"),

    # Requests to select the object
    matchedCuts = cms.string('coll("hltL1extraParticles")'),

    # 90% compatible with documentation at SWGuidePATTrigger#Module_Configuration_AN1
#    andOr          = cms.bool( False ), # if False, do the 'AND' of the conditions below; otherwise, do the OR
#    filterIdsEnum  = cms.vstring( '*' ),
#    filterIds      = cms.vint32( 0 ),
#    filterLabels   = cms.vstring( '*' ),
#    pathNames      = cms.vstring( '*' ),
#    collectionTags = cms.vstring( 'hltL1extraParticles' ),
    resolveAmbiguities    = cms.bool( True ), # if True, no more than one reco object can be matched to the same L1 object; precedence is given to the reco ones coming first in the list

    # Fake filter lavels for the object propagated to the second muon station
    setPropLabel = cms.string("propagatedToM2"),

    # Write extra ValueMaps
    writeExtraInfo = cms.bool(True),
)
