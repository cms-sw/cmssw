import FWCore.ParameterSet.Config as cms

def customise_HBHE_Method2(process):
   if hasattr(process,'hbheprereco'): 
      process.hbheprereco.puCorrMethod = cms.int32(2)
   return process
