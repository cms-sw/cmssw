import FWCore.ParameterSet.Config as cms


from DQM.HLTEvF.HLTObjectMonitorHeavyIon_cfi import *

hlt4vector = cms.Path(
    hltObjectMonitorHeavyIon
)
