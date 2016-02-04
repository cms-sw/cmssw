import FWCore.ParameterSet.Config as cms

pfIsolatedElectrons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedElectrons"),
    isolationValueMaps = cms.VInputTag(
        cms.InputTag("isoValElectronWithCharged"),
        cms.InputTag("isoValElectronWithNeutral"),
        cms.InputTag("isoValElectronWithPhotons")
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
    ## not used when isCombined=False
    # default value for combined relative with DR={0.4,0.4,0.4}
    # and weight={1.,1.,1.}; no optimal value found, yet.
    combinedIsolationCut = cms.double(0.2)
    )
                            
