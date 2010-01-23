import FWCore.ParameterSet.Config as cms

# Examples for configurations of the trigger match for various physics objects
#
# A detailed description is given in
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#PATTriggerMatcher
# Cuts on the parameters
# - 'maxDPtRel' and
# - 'maxDeltaR'
# are NOT tuned (using old values from TQAF MC match, January 2008)


## PAT Tuple (modified), 8E29 (start-up) ##

# matches to HLT_IsoMu3
muonTriggerMatchHLTIsoMu3 = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "cleanPatMuons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( '*' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_IsoMu3' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)

# matches to HLT_Mu3
muonTriggerMatchHLTMu3 = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "cleanPatMuons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( '*' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_Mu3' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)

# # matches to HLT_DoubleIsoMu3
# muonTriggerMatchHLTDoubleIsoMu3 = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
#     src     = cms.InputTag( "cleanPatMuons" ),
#     matched = cms.InputTag( "patTrigger" ),
#     andOr          = cms.bool( False ),
#     filterIdsEnum  = cms.vstring( '*' ),
#     filterIds      = cms.vint32( 0 ),
#     filterLabels   = cms.vstring( '*' ),
#     pathNames      = cms.vstring( 'HLT_DoubleIsoMu3' ),
#     collectionTags = cms.vstring( '*' ),
#     maxDPtRel = cms.double( 0.5 ),
#     maxDeltaR = cms.double( 0.5 ),
#     resolveAmbiguities    = cms.bool( True ),
#     resolveByMatchQuality = cms.bool( False )
# )

# matches to HLT_DoubleMu3
muonTriggerMatchHLTDoubleMu3 = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "cleanPatMuons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( '*' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_DoubleMu3' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)

# # matches to HLT_IsoEle15_LW_L1I
# electronTriggerMatchHLTIsoEle15LWL1I = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
#     src     = cms.InputTag( "cleanPatElectrons" ),
#     matched = cms.InputTag( "patTrigger" ),
#     andOr          = cms.bool( False ),
#     filterIdsEnum  = cms.vstring( '*' ),
#     filterIds      = cms.vint32( 0 ),
#     filterLabels   = cms.vstring( '*' ),
#     pathNames      = cms.vstring( 'HLT_IsoEle15_LW_L1I' ),
#     collectionTags = cms.vstring( '*' ),
#     maxDPtRel = cms.double( 0.5 ),
#     maxDeltaR = cms.double( 0.5 ),
#     resolveAmbiguities    = cms.bool( True ),
#     resolveByMatchQuality = cms.bool( False )
# )

# matches to HLT_Ele15_LW_L1R
electronTriggerMatchHLTEle15LWL1R = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "cleanPatElectrons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( '*' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_Ele15_LW_L1R' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)

# # matches to HLT_DoubleIsoEle10_LW_L1I
# electronTriggerMatchHLTDoubleIsoEle10LWL1I = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
#     src     = cms.InputTag( "cleanPatElectrons" ),
#     matched = cms.InputTag( "patTrigger" ),
#     andOr          = cms.bool( False ),
#     filterIdsEnum  = cms.vstring( '*' ),
#     filterIds      = cms.vint32( 0 ),
#     filterLabels   = cms.vstring( '*' ),
#     pathNames      = cms.vstring( 'HLT_DoubleIsoEle10_LW_L1I' ),
#     collectionTags = cms.vstring( '*' ),
#     maxDPtRel = cms.double( 0.5 ),
#     maxDeltaR = cms.double( 0.5 ),
#     resolveAmbiguities    = cms.bool( True ),
#     resolveByMatchQuality = cms.bool( False )
# )

# matches to HLT_DoubleEle5_SW_L1R
electronTriggerMatchHLTDoubleEle5SWL1R = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "cleanPatElectrons" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( '*' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_DoubleEle5_SW_L1R' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)

# # matches to HLT_LooseIsoTau_MET30_L1MET
# tauTriggerMatchHLTLooseIsoTauMET30L1MET = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
#     src     = cms.InputTag( "cleanPatTaus" ),
#     matched = cms.InputTag( "patTrigger" ),
#     andOr          = cms.bool( False ),
#     filterIdsEnum  = cms.vstring( '*' ),
#     filterIds      = cms.vint32( 0 ),
#     filterLabels   = cms.vstring( '*' ),
#     pathNames      = cms.vstring( 'HLT_LooseIsoTau_MET30_L1MET' ),
#     collectionTags = cms.vstring( '*' ),
#     maxDPtRel = cms.double( 0.5 ),
#     maxDeltaR = cms.double( 0.5 ),
#     resolveAmbiguities    = cms.bool( True ),
#     resolveByMatchQuality = cms.bool( False )
# )

# matches to HLT_DoubleLooseIsoTau15
tauTriggerMatchHLTDoubleLooseIsoTau15 = cms.EDFilter( "PATTriggerMatcherDRDPtLessByR",
    src     = cms.InputTag( "cleanPatTaus" ),
    matched = cms.InputTag( "patTrigger" ),
    andOr          = cms.bool( False ),
    filterIdsEnum  = cms.vstring( '*' ),
    filterIds      = cms.vint32( 0 ),
    filterLabels   = cms.vstring( '*' ),
    pathNames      = cms.vstring( 'HLT_DoubleLooseIsoTau15' ),
    collectionTags = cms.vstring( '*' ),
    maxDPtRel = cms.double( 0.5 ),
    maxDeltaR = cms.double( 0.5 ),
    resolveAmbiguities    = cms.bool( True ),
    resolveByMatchQuality = cms.bool( False )
)


## Sequences ##

# patTriggerMatcherPhoton = cms.Sequence(
# )
patTriggerMatcherElectron = cms.Sequence(
    electronTriggerMatchHLTEle15LWL1R      +
    electronTriggerMatchHLTDoubleEle5SWL1R
)
patTriggerMatcherMuon = cms.Sequence(
    muonTriggerMatchHLTIsoMu3    +
    muonTriggerMatchHLTMu3       +
    muonTriggerMatchHLTDoubleMu3
)
patTriggerMatcherTau = cms.Sequence(
    tauTriggerMatchHLTDoubleLooseIsoTau15
)
# patTriggerMatcherJet = cms.Sequence(
# )
# patTriggerMatcherMET = cms.Sequence(
# )

# patTriggerMatcher = cms.Sequence(
#     patTriggerMatcherPhoton   +
#     patTriggerMatcherElectron +
#     patTriggerMatcherMuon     +
#     patTriggerMatcherTau      +
#     patTriggerMatcherJet      +
#     patTriggerMatcherMET
# )
patTriggerMatcher = cms.Sequence(
    patTriggerMatcherElectron +
    patTriggerMatcherMuon     +
    patTriggerMatcherTau
)
