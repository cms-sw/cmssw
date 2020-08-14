import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripESProducers.SiStripConnectivity_cfi import (
    SiStripConnectivity as _SiStripConnectivity,
)

hltPhase2sistripconn = _SiStripConnectivity.clone()
