###############
# FastSim equivalent of RecoTracker/Configuration/python/RecoTracker_cff.py
###############

import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.MeasurementTrackerEventProducer_cfi import *

from FastSimulation.Tracking.iterativeTk_cff import *

# todo (long term) import dedx estimators here

# todo: import electron seeds here

# todo: import RecoTrackerNotStandard_cff

# todo : define same sequences as in RecoTracker/Configuration/python/RecoTracker_cff.py

from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
