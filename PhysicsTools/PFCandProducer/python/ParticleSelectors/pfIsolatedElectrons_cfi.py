import FWCore.ParameterSet.Config as cms

pfIsolatedElectrons  = cms.EDProducer(
    "IsolatedPFCandidateSelector",
    src = cms.InputTag("pfElectronsPtGt5"),
    isolationValueMaps = cms.VInputTag(
       cms.InputTag("pfElectronIsolationFromDepositsChargedHadrons"),
       #       cms.InputTag("pfElectronIsolationFromDepositsNeutralHadrons"),
       #       cms.InputTag("pfElectronIsolationFromDepositsPhotons")
       ),
    isolationCuts = cms.vdouble( 2.
#                                 1.,
#                                  1.
                                 )
    )
