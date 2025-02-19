import FWCore.ParameterSet.Config as cms

from RecoBTag.TrackProbability.trackProbabilityFrontierCond_cfi import *
trackProbabilityFrontierCond.connect = 'frontier://FrontierDev/CMS_COND_BTAU'

