import FWCore.ParameterSet.Config as cms

pfIsolatedPhotons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedPhotons"),
    isolationValueMapsCharged = cms.VInputTag(
        cms.InputTag("phPFIsoValueCharged04PFId"),
        ),
    isolationValueMapsNeutral = cms.VInputTag(
        cms.InputTag("phPFIsoValueNeutral04PFId"),
        cms.InputTag("phPFIsoValueGamma04PFId")
        ),
    doDeltaBetaCorrection = cms.bool(False),
    deltaBetaIsolationValueMap = cms.InputTag("phPFIsoValuePU04PFId"),
    deltaBetaFactor = cms.double(-0.5),    
    ## if True isolation is relative to pT
    isRelative = cms.bool(True),
    isolationCut = cms.double(999) 
    )
