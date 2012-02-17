import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitorClient.L1THcalClient_cfi import *
l1thcalseqClient = cms.Sequence(l1THcalClient)

