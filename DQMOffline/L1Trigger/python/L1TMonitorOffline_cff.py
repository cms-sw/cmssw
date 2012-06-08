import FWCore.ParameterSet.Config as cms

# adapt the L1TMonitor_cff configuration to offline DQM

# DQM online L1 Trigger modules 
from DQM.L1TMonitor.L1TMonitor_cff import *

# DTTF to offline configuration
l1tDttf.online = False

# input tag for BXTimining
bxTiming.FedSource = 'rawDataCollector'
