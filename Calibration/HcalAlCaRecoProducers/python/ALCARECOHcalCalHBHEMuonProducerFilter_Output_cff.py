import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL HBHEMuon
# output module 
#  module alcastreamHcalHBHEMuonOutput = PoolOutputModule
OutALCARECOHcalCalHBHEMuonProducerFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHBHEMuonProducerFilter')
        ),
    outputCommands = cms.untracked.vstring( 
        'keep *_alcaHcalHBHEMuonProducer_hbheMuon_*',
        )
)


import copy
OutALCARECOHcalCalHBHEMuonProducerFilter=copy.deepcopy(OutALCARECOHcalCalHBHEMuonProducerFilter_noDrop)
OutALCARECOHcalCalHBHEMuonProducerFilter.outputCommands.insert(0, "drop *")
