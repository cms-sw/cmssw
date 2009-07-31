import FWCore.ParameterSet.Config as cms

pfIsolatedMuons  = cms.EDProducer(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfMuonsPtGt5"),
    isolationValueMaps = cms.VInputTag(
       cms.InputTag("pfMuonIsolationFromDepositsChargedHadrons"),
       cms.InputTag("pfMuonIsolationFromDepositsNeutralHadrons"),
       cms.InputTag("pfMuonIsolationFromDepositsPhotons")
       ),
    isolationCuts = cms.vdouble(4,
                                2,
                                999.
                                )
    )
