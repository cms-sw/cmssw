import FWCore.ParameterSet.Config as cms

from RecoBTag.TrackProbability.trackProbabilityFrontierCond_cfi import *
trackProbabilityFrontierCond.connect = 'frontier://cms_conditions_data/CMS_COND_20X_BTAU'

