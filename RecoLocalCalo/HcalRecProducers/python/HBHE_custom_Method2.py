import FWCore.ParameterSet.Config as cms

def customise_HBHE_Method2(process):
   if hasattr(process,'hbheprereco'): 
      process.hbheprereco.puCorrMethod = cms.int32(0)
   return process
