import FWCore.ParameterSet.Config as cms

pfIsolatedMuons  = cms.EDProducer(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfMuonsPtGt5"),
    isolationValueMaps = cms.VInputTag(
        cms.InputTag("isoValMuonWithCharged"),
        cms.InputTag("isoValMuonWithNeutral"),
        cms.InputTag("isoValMuonWithPhotons")
        ),
    isolationCuts = cms.vdouble( 10,
                                 10,
                                 10 ),
    isolationCombRelIsoCut = cms.double(-1.0)
    )
