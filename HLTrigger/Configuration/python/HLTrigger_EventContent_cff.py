import FWCore.ParameterSet.Config as cms

#
# EventContent for HLT related products.
#
# This cff file exports the following five EventContent blocks:
#   HLTriggerRAW  HLTriggerRECO  HLTriggerAOD (without DEBUG products)
#   HLTDebugRAW   HLTDebugFEVT                (with    DEBUG products)
#
# as these are used in Configuration/EventContent
#
# All else is internal and should not be used directly by non-HLT users.
#
from HLTrigger.Configuration.HLTDefaultOutput_cff import *
from HLTrigger.Configuration.HLTDebugOutput_cff import *
from HLTrigger.Configuration.HLTDebugWithAlCaOutput_cff import *
# temporarily
HLTriggerFEVT = cms.PSet(
    outputCommands = cms.vstring()
)
# temporarily
HLTriggerRAW = cms.PSet(
    outputCommands = cms.vstring()
)
HLTriggerRECO = cms.PSet(
    outputCommands = cms.vstring()
)
HLTriggerAOD = cms.PSet(
    outputCommands = cms.vstring()
)
HLTDebugRAW = cms.PSet(
    outputCommands = cms.vstring()
)
HLTDebugFEVT = cms.PSet(
    outputCommands = cms.vstring()
)
HLTriggerFEVT.outputCommands.extend(block_hltDefaultOutput.outputCommands)
HLTriggerRAW.outputCommands.extend(block_hltDefaultOutput.outputCommands)
HLTriggerRECO.outputCommands.extend(block_hltDefaultOutput.outputCommands)
HLTriggerAOD.outputCommands.extend(block_hltDefaultOutput.outputCommands)
HLTDebugRAW.outputCommands.extend(block_hltDebugWithAlCaOutput.outputCommands)
HLTDebugFEVT.outputCommands.extend(block_hltDebugWithAlCaOutput.outputCommands)

