import FWCore.ParameterSet.Config as cms

L1TriggerKeyOnline = cms.ESProducer("L1TriggerKeyOnlineProd",
    subsystemLabels = cms.vstring( 'CSCTF', 'DTTF', 'RPC', 'GMT',
                                   'RCT', 'GCT', 'GT', 'TSP0' )
)


