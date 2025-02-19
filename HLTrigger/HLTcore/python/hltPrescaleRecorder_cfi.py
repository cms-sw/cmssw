import FWCore.ParameterSet.Config as cms

hltPrescaleRecorder = cms.EDProducer("HLTPrescaleRecorder",
    # (single) source:
    # -1:PrescaleServicePSet, 0:PrescaleService,
    #  1:Run, 2:Lumi, 3:Event, 4:CondDB    
    src         = cms.int32(0),
    # (multiple) destinations
    run         = cms.bool(True),
    lumi        = cms.bool(True),
    event       = cms.bool(True),
    condDB      = cms.bool(True),
    #                                     
    psetName    = cms.string(""),         # src=-1
    hltInputTag = cms.InputTag("","",""), # src= 1,2,3
    hltDBTag    = cms.string("")          # src= 4
)
