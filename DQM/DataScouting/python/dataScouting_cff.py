import FWCore.ParameterSet.Config as cms

from DQM.DataScouting.razorScouting_cff import *

#this file contains the sequence for data scouting
dataScoutingDQMSequence = cms.Sequence(scoutingRazorDQMSequence)

