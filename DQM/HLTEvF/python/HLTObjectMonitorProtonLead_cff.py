import FWCore.ParameterSet.Config as cms


from DQM.HLTEvF.HLTObjectMonitorProtonLead_cfi import *

hlt4vector = cms.Path(
    hltObjectMonitorProtonLead
)
