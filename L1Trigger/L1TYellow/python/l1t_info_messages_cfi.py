
#
# Sends INFO messages from L1T to cout
#
import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service(
    "MessageLogger",
    destinations       =  cms.untracked.vstring('cout'),
    categories         = cms.untracked.vstring('l1t', "yellow"),
    debugModules       = cms.untracked.vstring('*'),
    
    cout          = cms.untracked.PSet(
       threshold  = cms.untracked.string('INFO'),
       l1t = cms.untracked.PSet (
          limit = cms.untracked.int32(100)
       ),
 # If you want only messages from yellow trigger:       
 #     yellow = cms.untracked.PSet (
 #         limit = cms.untracked.int32(100)
 #     ),
       default = cms.untracked.PSet (
          limit = cms.untracked.int32(0)
       ),
       ERROR = cms.untracked.PSet(
          limit = cms.untracked.int32(100)
       )
    )
    )
