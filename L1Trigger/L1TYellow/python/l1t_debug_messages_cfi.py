#
# Send DEBUG messages from L1T to file l1t_debug.log
#
# NOTE:  to receive debug messages you must have compiled like this:
#
#   scram b -j 8 USER_CXXFLAGS="-DEDM_ML_DEBUG"
#
import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service(
    "MessageLogger",
    destinations       =  cms.untracked.vstring('cout', 'l1t_debug'),
    categories         = cms.untracked.vstring('l1t', "yellow"),
    debugModules       = cms.untracked.vstring('*'),
    
    cout          = cms.untracked.PSet(
       threshold  = cms.untracked.string('INFO'),
       l1t = cms.untracked.PSet (
          limit = cms.untracked.int32(100)
       ),
       default = cms.untracked.PSet (
          limit = cms.untracked.int32(0)
       ),
       ERROR = cms.untracked.PSet(
          limit = cms.untracked.int32(100)
       )
    ),
    l1t_debug          = cms.untracked.PSet(
       threshold =  cms.untracked.string('DEBUG'),
       default = cms.untracked.PSet (
          limit = cms.untracked.int32(0)
       ),
       l1t = cms.untracked.PSet (
          limit = cms.untracked.int32(100)
       )
    )
    )
