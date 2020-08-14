import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import (
    SiStripClusterChargeCutNone as _SiStripClusterChargeCutNone,
)

SiStripClusterChargeCutNone = _SiStripClusterChargeCutNone.clone()
