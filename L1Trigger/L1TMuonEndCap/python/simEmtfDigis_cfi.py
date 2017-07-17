import FWCore.ParameterSet.Config as cms

## EMTF emulator configuration
## Three options for CSCInput
##   * 'simCscTriggerPrimitiveDigis','MPCSORTED' : simulated trigger primitives (LCTs) from re-emulating CSC digis
##   * 'csctfDigis' : real trigger primitives as received by CSCTF (legacy trigger)
##   * 'emtfStage2Digis' : real trigger primitives as received by EMTF, unpacked in EventFilter/L1TRawToDigi/
## Two options for RPCInput
##   * 'simMuonRPCDigis' : simulated trigger primitives (LCTs) from emulating RPCs
##   * 'muonRPCDigis' : real trigger primitives from RPCs, but not unpacked by EMTF
## Two options for CSCInputBxShift
##   * -2 when using simCscTriggerPrimitiveDigis re-emulated from data
##   *  0 in all other cases (MC, csctfDigis or emtfStage2Digis input)
simEmtfDigis = cms.EDProducer("L1TMuonEndCapTrackProducer",
                              CSCInput = cms.InputTag('simCscTriggerPrimitiveDigis','MPCSORTED'),
                              RPCInput = cms.InputTag('simMuonRPCDigis'),
                              CSCInputBxShift = cms.untracked.int32(0),
                              )
