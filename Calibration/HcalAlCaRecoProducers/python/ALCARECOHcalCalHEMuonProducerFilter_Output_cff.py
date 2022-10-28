import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL HEMuon
# output module 
#  module alcastreamHcalHEMuonOutput = PoolOutputModule
OutALCARECOHcalCalHEMuonProducerFilter_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalHEMuonProducerFilter')
        ),
    outputCommands = cms.untracked.vstring( 
        'keep *_alcaHcalHBHEMuonProducer_hbheMuon_*',
        )
)


import copy
OutALCARECOHcalCalHEMuonProducerFilter=copy.deepcopy(OutALCARECOHcalCalHEMuonProducerFilter_noDrop)
OutALCARECOHcalCalHEMuonProducerFilter.outputCommands.insert(0, "drop *")
