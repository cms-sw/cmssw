import FWCore.ParameterSet.Config as cms

pfIsolatedMuons  = cms.EDProducer(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfMuonsPtGt5"),
    isolationValueMaps = cms.VInputTag(
       cms.InputTag("isoValMuonWithCharged"),
       cms.InputTag("isoValMuonWithNeutral"),
       cms.InputTag("isoValMuonWithPhotons")
       ),
    isolationCuts = cms.vdouble(99999.,
                                99999.,
                                99999. ),
    isolationCombRelIsoCut = cms.double(0.14)
    )
