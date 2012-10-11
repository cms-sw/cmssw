import FWCore.ParameterSet.Config as cms

from DQM.DataScouting.razorScouting_cff import *
from DQM.DataScouting.dijetScouting_cff import *
from DQM.DataScouting.alphaTScouting_cff import *

#this file contains the sequence for data scouting
dataScoutingDQMSequence = cms.Sequence(scoutingRazorDQMSequence*scoutingDiJetDQMSequence*scoutingAlphaTDQMSequence)

