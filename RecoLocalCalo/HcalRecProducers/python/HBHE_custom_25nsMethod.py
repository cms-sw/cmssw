import FWCore.ParameterSet.Config as cms

def customise_HBHE_Method1(process):
   if hasattr(process,'hbheprereco'): 
      process.hbheprereco.puCorrMethod = cms.int32(1)
   return process

def customise_HBHE_Method0(process):
   if hasattr(process,'hbheprereco'): 
      process.hbheprereco.puCorrMethod = cms.int32(0)
   return process

