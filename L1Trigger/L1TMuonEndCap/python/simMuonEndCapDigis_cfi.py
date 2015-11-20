import FWCore.ParameterSet.Config as cms

###EMTF emulator configuration
simEmtfDigis = cms.EDProducer("L1TMuonEndCapTrackProducer",
                              CSCInput = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED')
                              )


