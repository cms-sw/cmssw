import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TDTTPG_cfi import *
#        module l1tdttpgpack = DTTFFEDSim{
#        }
l1tdttpgunpack = cms.EDProducer("DTTFFEDReader")

l1tdttpgpath = cms.Path(l1tdttpgunpack*l1tdttpg)

