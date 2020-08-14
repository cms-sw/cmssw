import FWCore.ParameterSet.Config as cms

from RecoTracker.TkNavigation.NavigationSchoolESProducer_cfi import (
    navigationSchoolESProducer as _navigationSchoolESProducer,
)

hltPhase2navigationSchoolESProducer = _navigationSchoolESProducer.clone()
