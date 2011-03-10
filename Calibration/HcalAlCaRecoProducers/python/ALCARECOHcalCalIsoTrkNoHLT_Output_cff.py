import FWCore.ParameterSet.Config as cms

# output block for alcastream HCAL Isotrk
# output module 
#  module alcastreamHcalIsotrkOutput = PoolOutputModule
OutALCARECOHcalCalIsoTrkNoHLT_noDrop = cms.PSet(
    # use this in case of filter available
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalIsoTrkNoHLT')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_IsoProd_*_*')
)


import copy
OutALCARECOHcalCalIsoTrkNoHLT=copy.deepcopy(OutALCARECOHcalCalIsoTrkNoHLT_noDrop)
OutALCARECOHcalCalIsoTrkNoHLT.outputCommands.insert(0, "drop *")
