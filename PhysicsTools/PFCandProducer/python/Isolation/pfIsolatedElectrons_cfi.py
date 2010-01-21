import FWCore.ParameterSet.Config as cms

pfIsolatedElectrons  = cms.EDProducer(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfElectronsPtGt5"),
    isolationValueMaps = cms.VInputTag(
        cms.InputTag("isoValElectronWithCharged"),
        cms.InputTag("isoValElectronWithNeutral"),
        cms.InputTag("isoValElectronWithPhotons")
       ),
    # no cut on the photon deposits yet
    isolationCuts = cms.vdouble( 10,
                                 10,
                                 10 ),
    isolationCombRelIsoCut = cms.double(-1.0)
    )
