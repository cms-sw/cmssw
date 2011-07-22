import FWCore.ParameterSet.Config as cms

pfIsolatedPhotons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedPhotons"),
    isolationValueMaps = cms.VInputTag(
        cms.InputTag("isoValPhotonWithCharged"),
        cms.InputTag("isoValPhotonWithNeutral"),
        cms.InputTag("isoValPhotonWithPhotons")
        ),
    ## if True isolation is relative to pT
    isRelative = cms.bool(True),
    ## if True all isoValues are combined (summed)
    isCombined = cms.bool(True),
    ## not used when isCombined=True
    # non-optimised default for loose absulute isolation
    isolationCuts = cms.vdouble( 10, 
                                 10,
                                 10 ),
    # not used when isCombined=False
    # default value for combined relative with DR={0.4,0.4,0.4}
    # and weight={1.,1.,1.}; optimised for Z->mu,mu
    combinedIsolationCut = cms.double(999) 
    )
