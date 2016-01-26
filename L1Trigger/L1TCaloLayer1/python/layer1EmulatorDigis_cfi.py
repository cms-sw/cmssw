import FWCore.ParameterSet.Config as cms

layer1EmulatorDigis = cms.EDProducer('L1TCaloLayer1',
                                     ecalTPSource = cms.InputTag("l1tCaloLayer1Digis"),
                                     hcalTPSource = cms.InputTag("l1tCaloLayer1Digis"),
                                     useLSB = cms.bool(True),
                                     useLUT = cms.bool(True),
                                     verbose = cms.bool(False)
                                     )
