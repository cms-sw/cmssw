import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

EgammaPostVal = DQMEDHarvester("EmDQMPostProcessor",
   subDir = cms.untracked.string("HLT/HLTEgammaValidation"),
   dataSet = cms.untracked.string("unknown"),                  
   noPhiPlots = cms.untracked.bool(True),                  
                              )
# foo bar baz
# cIbxXf90WpFIf
# 0XISF123YeYuX
