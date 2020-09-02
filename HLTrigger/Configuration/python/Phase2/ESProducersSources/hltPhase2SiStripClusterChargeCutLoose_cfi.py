import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import (
    SiStripClusterChargeCutLoose as _SiStripClusterChargeCutLoose,
)

SiStripClusterChargeCutLoose = _SiStripClusterChargeCutLoose.clone()
