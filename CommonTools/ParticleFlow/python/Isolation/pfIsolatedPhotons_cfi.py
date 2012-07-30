import FWCore.ParameterSet.Config as cms

pfIsolatedPhotons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedPhotons"),
    isolationValueMapsCharged = cms.VInputTag(
        cms.InputTag("phPFIsoValueCharged04"),
        ),
    isolationValueMapsNeutral = cms.VInputTag(
        cms.InputTag("phPFIsoValueNeutral04"),
        cms.InputTag("phPFIsoValueGamma04")
        ),
    doDeltaBetaCorrection = cms.bool(False),
    deltaBetaIsolationValueMap = cms.InputTag("phPFIsoValuePU04"),
    deltaBetaFactor = cms.double(-0.5),    
    ## if True isolation is relative to pT
    isRelative = cms.bool(True),
    isolationCut = cms.double(999) 
    )
