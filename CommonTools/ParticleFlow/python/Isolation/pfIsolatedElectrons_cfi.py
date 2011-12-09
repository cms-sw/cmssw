import FWCore.ParameterSet.Config as cms

pfIsolatedElectrons  = cms.EDFilter(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfSelectedElectrons"),
    isolationValueMapsCharged = cms.VInputTag(
        cms.InputTag("isoValElectronWithCharged"),
       ),
    isolationValueMapsNeutral = cms.VInputTag(
        cms.InputTag("isoValElectronWithNeutral"),
        cms.InputTag("isoValElectronWithPhotons")
       ),
    doDeltaBetaCorrection = cms.bool(False),
    deltaBetaIsolationValueMap = cms.InputTag(""),
    deltaBetaFactor = cms.double(-0.5),    
    ## if True isolation is relative to pT
    isRelative = cms.bool(True),
    isolationCut = cms.double(0.2)
    )
                            
