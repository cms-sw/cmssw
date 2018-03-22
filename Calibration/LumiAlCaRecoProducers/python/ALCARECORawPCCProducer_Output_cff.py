import FWCore.ParameterSet.Config as cms


OutALCARECORawPCCProducer_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECORawPCCProducer')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_rawPCCProd_*_*')
        )


import copy

OutALCARECORawPCCProducer=copy.deepcopy(OutALCARECORawPCCProducer_noDrop)
OutALCARECORawPCCProducer.outputCommands.insert(0, "drop *")
