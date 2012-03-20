import FWCore.ParameterSet.Config as cms

pfIsolatedElectrons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedElectrons"),
    isolationValueMapsCharged = cms.VInputTag(
        cms.InputTag("elPFIsoValueCharged04PFId"),
       ),
    isolationValueMapsNeutral = cms.VInputTag(
        cms.InputTag("elPFIsoValueNeutral04PFId"),
        cms.InputTag("elPFIsoValueGamma04PFId")
       ),
    doDeltaBetaCorrection = cms.bool(False),
    deltaBetaIsolationValueMap = cms.InputTag("elPFIsoValuePU04PFId"),
    deltaBetaFactor = cms.double(-0.5),    
    ## if True isolation is relative to pT
    isRelative = cms.bool(True),
    isolationCut = cms.double(0.2)
    )
                            
