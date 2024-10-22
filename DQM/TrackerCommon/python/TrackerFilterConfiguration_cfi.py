import FWCore.ParameterSet.Config as cms
#-----------------------------------
#  Detector State Filter
#-----------------------------------
import DQM.TrackerCommon._detectorStateFilter_cfi as _dcs
detectorStateFilter = _dcs._detectorStateFilter.clone()
#-----------------------------------
#  Simple Event Filter
#-----------------------------------
import DQM.TrackerCommon._simpleEventFilter_cfi  as _sef
simpleEventFilter = _sef._simpleEventFilter.clone()
