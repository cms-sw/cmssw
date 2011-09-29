import FWCore.ParameterSet.Config as cms

pfIsolatedPhotons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedPhotons"),
    isolationValueMapsCharged = cms.VInputTag(
        cms.InputTag("isoValPhotonWithCharged"),
        ),
    isolationValueMapsNeutral = cms.VInputTag(
        cms.InputTag("isoValPhotonWithNeutral"),
        cms.InputTag("isoValPhotonWithPhotons")
        ),
    doDeltaBetaCorrection = cms.bool(False),
    deltaBetaIsolationValueMap = cms.InputTag(""),
    deltaBetaFactor = cms.double(-0.5),    
    ## if True isolation is relative to pT
    isRelative = cms.bool(True),
    isolationCut = cms.double(999) 
    )
