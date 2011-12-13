import FWCore.ParameterSet.Config as cms


pfIsolatedMuons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedMuons"),
    isolationValueMapsCharged = cms.VInputTag(
        cms.InputTag("muPFIsoValueCharged04"),
        ),
    isolationValueMapsNeutral =  cms.VInputTag(
        cms.InputTag("muPFIsoValueNeutral04"),
        cms.InputTag("muPFIsoValueGamma04")
        ),
    doDeltaBetaCorrection = cms.bool(False),
    deltaBetaIsolationValueMap = cms.InputTag("muPFIsoValuePU04"),
    deltaBetaFactor = cms.double(-0.5),
    ## if True isolation is relative to pT
    isRelative = cms.bool(True),
    ## if True all isoValues are combined (summed)
    # isCombined = cms.bool(True),
    ## not used when isCombined=True
    # non-optimised default for loose absulute isolation
    # isolationCuts = cms.vdouble( 10, 
    #                             10,
    #                             10 ),
    # not used when isCombined=False
    # default value for combined relative with DR={0.4,0.4,0.4}
    # and weight={1.,1.,1.}; optimised for Z->mu,mu
    isolationCut = cms.double(0.15) 
    )
